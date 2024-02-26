import json
import argparse
import logging 
import argparse
from pathlib import Path
from os.path import exists
import os
import torch
from typing import Any, Dict, List, Union
import numpy as np
from my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification
from transformers import AutoConfig, AutoTokenizer, set_seed, PretrainedConfig
from datasets import load_dataset
from tqdm import tqdm
import pickle as pkl
from scipy.optimize import LinearConstraint, minimize
import random

IGNORE_TAG = "Ignore"
NO_ERROR_TAG = "O"
ERROR_TAG = "ERR"

MAX_PASSES = 2
EPSILON = 0.000001
# EPSILON = 0
BIG_NUM = 225
MAX_TRIES = 5

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-data",
    type=str,
    help="Path to file containing data"
)

parser.add_argument(
    "-model",
    type=str,
    help="Path to Longformer model on huggingface",
    default="allenai/longformer-base-4096",
)

parser.add_argument(
    '-seed', 
    type=int, 
    help='Random seed', 
    default=11892
)

parser.add_argument(
    "-lm_output_column_prefix",
    type=str, 
    help="The column name of text to input in the file (a csv or JSON file).",
    default="prediction"
)

parser.add_argument(
    "-preference_column",
    type=str, 
    help="The column name of text to input in the file (a csv or JSON file).",
    default="preference"
)

parser.add_argument(
    "-n_lm_outputs",
    type=int, 
    help="The number of LM outputs being generated for each input, included in the json files.",
    default=4
)

parser.add_argument(
    "-max_seq_length",
    type=int, 
    help=(
        "The maximum total input sequence length after tokenization. If set, sequences longer "
        "than this will be truncated, sequences shorter will be padded."
    ),
    default=2048
)

parser.add_argument(
    "-pad_to_max_length",
    action="store_true",
    help=(
        "Whether to pad all samples to model maximum sentence length. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
        "efficient on GPU but very bad for TPU."
    )
)

parser.add_argument(
    "-loadFeats",
    action="store_true",
    help="Boolean flag to load features from features.pkl",
)

parser.add_argument(
    "-factuality",
    action="store_true",
    help="Boolean flag to use factuality data (instead of preference data)"
)

parser.add_argument(
    "-addEpsilon",
    action="store_true",
    help="Boolean flag to add epsilon_i for every constraint"
)
#----------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        raise ValueError("Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")
#----------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#----------------------------------------------------------------------
def readFile(filePath):
    data = []
    if filePath.endswith(".json"):
        with open(filePath, "r") as f:
            data = list(json.load(f))
    elif filePath.endswith(".jsonl"):
        with open(filePath, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line))
    else: 
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")
    return data
#----------------------------------------------------------------------
def writeFile(filePath, data):
    if filePath.endswith(".json"):
        with open(filePath, "w") as f:
            json.dump(data, f)
    elif filePath.endswith(".jsonl"):
        with open(filePath, "w") as f:
            for d in data: 
                f.write(json.dumps(d))
                f.write("\n")
    else: 
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")
#----------------------------------------------------------------------
def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    checkFile(args.data)

    # Set seed before initializing model.
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    features = []
    if args.loadFeats:
        if args.factuality:
            with open("features_factuality.pkl","rb") as f:
                features = pkl.load(f) 
        else: 
            with open("features.pkl","rb") as f:
                features = pkl.load(f) 
    else:
        # Read the datasets
        data_files = {}
        data_files["train"] = args.data
        extension = args.data.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            args.model,
            # num_labels=num_labels,
        )

        if args.factuality:
            model = LongformerForTokenClassification.from_pretrained(
                args.model,
                from_tf=bool(".ckpt" in args.model),
                config=config,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                add_prefix_space=True,
            )
        else: 
            model = LongformerForSequenceClassification.from_pretrained(
                args.model,
                from_tf=bool(".ckpt" in args.model),
                config=config,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
            )

        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if args.pad_to_max_length else False

        def process_annotations(annotations):
            examples = []

            prediction_keys = [f"{args.lm_output_column_prefix} {i+1}" for i in range(args.n_lm_outputs)]
            n_comparisons = (args.n_lm_outputs * (args.n_lm_outputs - 1)) // 2

            for ann in tqdm(annotations, desc="Processing annotations"):    
                predictions = [ann[key] for key in prediction_keys]

                # construct prompt
                tokens = ["question:"] + ann["question"].strip().split()
                for i, p in enumerate(ann['passages']):
                    if i == 0:
                        tokens += ["context:"]
                    title_string = f"wikipage: {p[0]}"
                    p_tokens = title_string.split()
                    p_tokens += ['text:'] + ' '.join(p[1:]).split()
                    tokens += p_tokens
                tokens += ['answer:']
                prompt = ' '.join(tokens)

                preferences = ann[args.preference_column]

                assert len(preferences) == n_comparisons

                pair_id = 0

                for i in range(len(predictions)):
                    for j in range(i+1, len(predictions)):
                        pref = preferences[pair_id] # 0 equal, 1 first better, 2 second better
                        pair_id += 1
                        example = {}
                        if pref == 0:
                            continue
                        elif pref == 1:
                            example["pred1"] = tokenizer.encode(
                                prompt + ' ' + predictions[i], 
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["pred2"] = tokenizer.encode(
                                prompt + ' ' + predictions[j],
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["label"] = 1
                        elif pref == 2:
                            example["pred1"] = tokenizer.encode(
                                prompt + ' ' + predictions[j],
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["pred2"] = tokenizer.encode(
                                prompt + ' ' + predictions[i],
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["label"] = 1
                        else:
                            raise("unknown preference")

                        examples.append(example)

            return examples
        
        def tokenize_and_align_labels(examples, input_column="text"):
            proInp = {
                "text": {},
                "corrected_pred": {}
            }
            tokenized_inputs = tokenizer(
                examples[input_column],
                padding=padding,
                truncation=True,
                max_length=args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )

            tokenized_inputs_corrected_pred = tokenizer(
                examples[input_column+"_corrected_pred"],
                padding=padding,
                truncation=True,
                max_length=args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )

            proInp["text"] = tokenized_inputs
            proInp["corrected_pred"] = tokenized_inputs_corrected_pred

            return proInp
    
        if args.factuality:
            train_dataset = raw_datasets["train"].map(
                tokenize_and_align_labels,
                batched=False,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        else:
            train_dataset = process_annotations(raw_datasets["train"])

        if args.factuality:
            with open("train_dataset_factuality.pkl", "wb") as f:
                pkl.dump(train_dataset, f)
        else:
            with open("train_dataset.pkl", "wb") as f:
                pkl.dump(train_dataset, f)

        model.eval()
        with torch.no_grad():
            if args.factuality:
                for d in tqdm(train_dataset, desc="Featurizing dataset"):
                    feat_1 = model(
                        input_ids=torch.tensor(d["corrected_pred"]["input_ids"]).unsqueeze(0),
                        attention_mask=torch.tensor(d["corrected_pred"]["attention_mask"]).unsqueeze(0)
                    )
                    feat_2 = model(
                        input_ids=torch.tensor(d["text"]["input_ids"]).unsqueeze(0),
                        attention_mask=torch.tensor(d["text"]["attention_mask"]).unsqueeze(0)
                    )
                    f1 = feat_1["sequence_output"].squeeze(0).sum(0)
                    f2 = feat_2["sequence_output"].squeeze(0).sum(0)
                    features.append(f1-f2) 
                    logging.info("Shape: {}".format(features[-1].shape))
                features = torch.stack(features)
                logging.info("Shape: {}".format(features.shape))
            else:
                for d in tqdm(train_dataset, desc="Featurizing dataset"):
                    feat_1 = model(d["pred1"])
                    feat_2 = model(d["pred2"])
                    features.append(feat_1["phi"].squeeze()-feat_2["phi"].squeeze()) 
                    logging.info("Shape: {}".format(features[-1].shape))
                features = torch.stack(features)
                logging.info("Shape: {}".format(features.shape))

        if args.factuality:
            with open("features_factuality.pkl", "wb") as f:
                pkl.dump(features, f)
        else:
            with open("features.pkl", "wb") as f:
                pkl.dump(features, f)

    non_redundant_inds = np.arange(len(features))
    redundant_inds = []
    failure_inds = []

    def objFunc(x, *funcArgs):
        if args.addEpsilon:
            return x[:funcArgs[0].shape[0]].dot(funcArgs[0]) + sum(x[funcArgs[0].shape[0]:])
        else:
            return x.dot(funcArgs[0])
    
    numPass = 0

    while(numPass < MAX_PASSES):
        indsToTest = non_redundant_inds.copy()
        logging.info("Pass {}/{}".format(numPass+1, MAX_PASSES))
        numPass += 1
        numRedundant = 0
        numFailure = 0
        for i, ind in tqdm(enumerate(indsToTest), desc="Checking for redundancy"):
            if i%25 == 0: 
                logging.info("Stats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(i, numRedundant, i-numRedundant-numFailure, numFailure))
            numTries = 0
            while 1:
                numTries += 1
                if args.addEpsilon:
                    constraints = torch.zeros((features.shape[0]-1, (features[0].shape[0]+features.shape[0])))
                    featuresConstraints = features[np.delete(non_redundant_inds, np.where(non_redundant_inds == ind))]
                    constraints[:featuresConstraints.shape[0], :featuresConstraints.shape[1]] = featuresConstraints
                else: 
                    constraints = features[np.delete(non_redundant_inds, np.where(non_redundant_inds == ind))]

                if args.addEpsilon:
                    # initialGuess = torch.zeros((features[0].shape[0]+features.shape[0],))
                    # initialGuess = BIG_NUM+torch.zeros((features[0].shape[0]+features.shape[0],))
                    initialGuess = torch.rand((features[0].shape[0]+features.shape[0],))
                else:
                    # initialGuess = torch.zeros(features[0].shape)
                    # initialGuess = BIG_NUM+torch.zeros(features[0].shape)
                    initialGuess = torch.rand(features[0].shape)

                m = minimize(
                    fun=objFunc,
                    x0=initialGuess,
                    constraints=LinearConstraint(
                        A=(-1)*constraints,
                        lb=(-1)*torch.inf,
                        ub=EPSILON,
                    ),
                    args=(features[ind].squeeze(), )
                )
                if not np.isclose(m.x[:features[0].shape[0]], torch.zeros(features[i].shape)).all():
                    break 
                else: 
                    if numTries > MAX_TRIES:
                        logging.info("Zero solution obtained. Execeeded MAX_TRIES. Exiting...")
                        m.success = False
                        break
                    logging.info("Zero solution obtained. Rerunning...")
            if numTries > MAX_TRIES or not m.success:
                numFailure += 1
                logging.info("Inconsistent constraints @ {}".format(ind))
                failure_inds.append(ind)
                continue 
            # if np.isclose(m.x[:features[0].shape[0]].dot(features[i]), 0):#Redundant
            if m.x[:features[0].shape[0]].dot(features[i]) >= 0:#Redundant
                numRedundant += 1
                logging.info("Redundancy @ {}".format(ind))
                redundant_inds.append(ind)
                non_redundant_inds = np.delete(non_redundant_inds, np.where(non_redundant_inds == ind))
            else: #Non-redundant
                pass 
        
    fileAddendum = []
    if args.addEpsilon: 
        fileAddendum.append("epsilon")
    if args.factuality:
        fileAddendum.append("factuality")

    fileEnding = "_".join(fileAddendum)

    with open(f"redundant{fileEnding}.pkl", "wb") as f: 
        pkl.dump(redundant_inds, f)
    
    with open(f"non_redundant{fileEnding}.pkl", "wb") as f: 
        pkl.dump(non_redundant_inds, f)
    
    with open(f"failure{fileEnding}.pkl", "wb") as f: 
        pkl.dump(failure_inds, f)
#----------------------------------------------------------------------
if __name__ == "__main__":
    main()
import argparse
import logging
import os
import openai
import random
import torch
import numpy as np
from pathlib import Path
from os.path import exists
import os
import json
import regex as re
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from difflib import SequenceMatcher as SM
from nltk.util import ngrams
import random

SAMPLES_TO_ANNOTATE_START = 0
SAMPLES_TO_ANNOTATE_END = 5

NUM_ROUNDS = {
    "R": 1,
    "F": 1,
    "C": 1,
}

RELEVANCE_CATEGORIES = ["Irrelevant", "Incoherent", "Redundant"]
FACTUALITY_CATEGORIES = ["Unverifiable", "Wrong-Grounding"]

STR_2_ETYPE = {
    "Missing Answer": "Missing-Answer",
    "Missing Major Auxiliary Information": "Missing-Major-Auxiliary",
    "Missing Minor Auxiliary Information": "Missing-Minor-Auxiliary",
}

SAVE_EVERY_N_ANNOTATIONS = 5

visualizeToEtypes = {
    "relevance": RELEVANCE_CATEGORIES,
    "factuality": FACTUALITY_CATEGORIES,
}

ANNOTATION_ORDER = {
    "RFC": ["relevance", "factuality", "completeness"],
    "RCF": ["relevance", "completeness", "factuality"],
    "FRC": ["factuality", "relevance", "completeness"], 
    "FCR": ["relevance", "completeness", "factuality"],
    "CRF": ["completeness", "relevance", "factuality"],
    "CFR": ["completeness", "factuality", "relevance"],
    "RF": ["relevance", "factuality"],
    "FR": ["factuality", "relevance"],
    "RC": ["relevance", "completeness"],
    "CR": ["completeness", "relevance"],
    "FC": ["factuality", "completeness"],
    "CF": ["completeness", "factuality"],
    "R": ["relevance"],
    "F": ["factuality"],
    "C": ["completeness",],
}

PROMPTS = {
"base": """Passages:
{passages}
    
Input Question:
{question}

Correct Response:
{referenceResponse}

Model Prediction:
{response}
""",
"relevance": {
    "CoT":"""######
Mark the irrelevant, incoherent and repetitive spans in the model prediction by enclosing them within square brackets ([]) and then provide an explanation.

Annotated Model Prediction:{annotation}
""",
    "noCoT":"""######
Mark the irrelevant, incoherent and repetitive spans in the model prediction by enclosing them within square brackets ([]).

Annotated Model Prediction:{annotation}
""",
#     "addendum":"""######
# Input Question:
# {question}

# Correct Response:
# {referenceResponse}

# Model Prediction:
# {response}
# """
},

"factuality": {
    "CoT":"""######
Mark the spans in the model prediction that contain inconsistent or unverifiable facts by listing the sentences by their sentence numbers and then provide an explanation.

Model Prediction Sentences:
{responseSentences}

Model Prediction Sentences:{annotation}
""",
    "noCoT":"""######
Mark the spans in the model prediction that contain inconsistent or unverifiable facts by listing the sentences by their sentence numbers.

Model Prediction Sentences:
{responseSentences}

Model Prediction Sentences:{annotation}
""",
#     "addendum":"""######
# {passages}

# Model Prediction:
# {response}
# """,
# "addendum":"""######
# {passages}
    
# Input Question:
# {question}

# Correct Response:
# {referenceResponse}

# Model Prediction:
# {response}
# """
},

"completeness": {
    "CoT":"""######
Mark missing information in the model prediction by listing out the passage and sentence numbers containing the missing information and then provide an explanation.

Missing Info:{annotation}
""",
    "noCoT":
    """######
Mark missing information in the model prediction by listing out the passage and sentence numbers containing the missing information.

Missing Info:{annotation}
""", 
#     "addendum":"""######
# {passages}
    
# Input Question:
# {question}

# Correct Response:
# {referenceResponse}

# Model Prediction:
# {response}
# """
},
"corrected-prediction": """######
Now, write the corrected model prediction addressing all mistakes labeled above. 

Corrected Model Prediction:{annotation}""",
}

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
    "-seed",
    type=int,
    help="Seed for torch/numpy",
    default=13
)

parser.add_argument(
    "-model",
    choices=["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    help="Name of OpenAI model to use",
    default="gpt-4"
)

parser.add_argument(
    "-temperature",
    type=float,
    help="Temperature for generations",
    default=0,
)

parser.add_argument(
    "-max_tokens",
    type=int,
    help="Max no. of tokens to generate",
    default=512,
)

parser.add_argument(
    "-top_p",
    type=float,
    help="top p for nucleus sampling",
    default=1,
)

parser.add_argument(
    "-frequency_penalty",
    type=float,
    help="Penalty for repeating lines verbatim",
    default=0,
)

parser.add_argument(
    "-presence_penalty",
    type=float,
    help="Penalty for repeating tokens",
    default=0,
)

parser.add_argument(
    "-systemPrompt",
    type=str,
    help="Path to directory containing system prompts [File with the name order_[CoT/noCoT].txt in this directory will be used]",
    default="./systemPrompts/"
)

parser.add_argument(
    "-data",
    type=str,
    help="Path to JSON file containing data",
    # default="./train_feedback.json"
    default="/uufs/chpc.utah.edu/common/home/u1419542/scratch/FineGrainedRLHF/tasks/qa_feedback/data/sampled/m250_n1000+250/rm/train_feedback.json"
)

parser.add_argument(
    "-dataStart",
    type=int, 
    help="Index for data instance to begin annotation from",
    default=SAMPLES_TO_ANNOTATE_START
)

parser.add_argument(
    "-dataEnd",
    type=int, 
    help="Index for data instance to end annotation from",
    default=SAMPLES_TO_ANNOTATE_END
)

parser.add_argument(
    "-dataNum",
    type=int, 
    help="No. of data instances to annotate; Instances are sampled randomly without replacement",
    default=-1
)

parser.add_argument(
    "-fewShot",
    type=str,
    help="Path to JSON file containing few shot demonstration",
    default="./dev_feedback.json"
)

parser.add_argument(
    "-numShots",
    type=int,
    help="No. of few shots to include in system prompt",
    default=0
)

parser.add_argument(
    "-out",
    type=str,
    help="Path to store annotations as a json file with the same name as the input to -data",
    # default="./annotations/"
    default="./annotations/m250_n1000+250/"
)

parser.add_argument(
    "-append",
    action="store_true",
    help="Boolean flag to append to preexisiting annotations of the same order",
)

parser.add_argument(
    "-evalOnly",
    action="store_true",
    help="Boolean flag to perform only evaluation on old annotations",
)

parser.add_argument(
    "-order",
    choices=list(ANNOTATION_ORDER.keys()),
    help="Order in which annotations are to be carried out",
    default="RFC"
)

parser.add_argument(
    "-numRoundsR",
    type=int,
    help="Number of rounds for relevance",
    default=NUM_ROUNDS["R"]
)

parser.add_argument(
    "-numRoundsF",
    type=int,
    help="Number of rounds for factuality",
    default=NUM_ROUNDS["F"]
)

parser.add_argument(
    "-numRoundsC",
    type=int,
    help="Number of rounds for completeness",
    default=NUM_ROUNDS["C"]
)

parser.add_argument(
    "-CoT",
    action="store_true",
    help="Boolean flag to enable CoT prompting"
)

parser.add_argument(
    "-visualize",
    choices=list(visualizeToEtypes.keys()),
    help="Visualize annotations; When this is set, only visualization would be performed; Only works when run directly from terminal"
)

parser.add_argument(
    "-saveFreq",
    type=int,
    help="Save frequency",
    default=SAVE_EVERY_N_ANNOTATIONS
)
#---------------------------------------------------------------------------
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
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None, returnBool=False):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            if returnBool: 
                return False
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        if returnBool: 
            return False
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        if returnBool: 
            return False
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
    if returnBool: 
        return True
#---------------------------------------------------------------------------
def readFile(fileName, error="readFile"):
    data = []
    fileExt = fileName.split(".")[-1]
    if fileExt == "txt":
        with open(fileName, "r") as f: 
            data = list(f.readlines())
            data = list(map(str.strip, data))
    elif fileExt == "json":
        with open(fileName, "r")  as f: 
            data = list(json.load(f)) 
    else: 
        raise ValueError(f"[{error}] Unsupported file type: {fileExt}")
    return data
#---------------------------------------------------------------------------
def writeFile(data, fileName: str, error: str="writeFile"):
    fileExt = fileName.split(".")[-1]
    if fileExt == "txt":
            with open(fileName, "w") as f: 
                for d in data:
                    f.write(d)
                    f.write("\n")
    elif fileExt == "json":
        with open(fileName, "w")  as f: 
            json.dump(data, f)
    else: 
        raise ValueError(f"[{error}] Unsupported file type: {fileExt}")
    return data
#---------------------------------------------------------------------------
def markErr(text, feedback, eTypes=["Irrelevant"]):
    spans = []
    for f in feedback["errors"]:
        if f["error type"] in eTypes:
            spans.append((f["start"], f["end"]))
    spans.sort()
    if len(spans) == 0:
        return text
    markedText = text[:spans[0][0]]
    for i in range(len(spans)):
        if i: 
            markedText += text[spans[(i-1)][1]:(spans[i][0]-1)]
        markedText += "[" + text[spans[i][0]:spans[i][1]] + "]"
    markedText += text[spans[-1][1]:]
    return markedText
#---------------------------------------------------------------------------
def markSents(text, feedback, eTypes=["Unverifiable"]):
    erroneousSents = set({})
    sentences = sent_tokenize(text)
    sentenceBoundaries = []
    for sent in sentences:
        sInd = text.index(sent)
        sentenceBoundaries.append((sInd, sInd+len(sent)))
    for g in feedback["errors"]:
        if g["error type"] in eTypes:
            for sbInd, sb in enumerate(sentenceBoundaries):
                if (sb[0] <= g["start"] <= sb[1]) or (sb[0] <= g["end"] <= sb[1]):
                    erroneousSents.add(sbInd)
    erroneousSents = list(erroneousSents)
    erroneousSents.sort()
    markedText = ""
    for i, es in enumerate(erroneousSents):
        markedText += "{}. S{}\n".format(i+1, es+1)
    if len(markedText) == 0:
        markedText = "None"
    return markedText
#---------------------------------------------------------------------------
def getMissingSentences(instanceFeedback, passID=None):
    out = []
    for i in range(len(instanceFeedback["missing-info"])):
        pId = instanceFeedback["missing-info"][i]["passage_id"]
        #When passID is passed, only that passage is considered
        if passID != None and pId != passID:
            continue
        for sId in instanceFeedback["missing-info"][i]["sentence_id"]:
            out.append("{}. Passage {}, sentence {} [{}]".format(len(out)+1, pId, sId, instanceFeedback["missing-info"][i]["error type"]))
    if len(out) == 0:
        out = ["None"]
    return "\n".join(out)
#---------------------------------------------------------------------------
def extractUnfactualSentences(text, markedText, eType="Irrelevant"):
    errors = []
    patt = '[0-9]*\\. S[0-9]*'
    lines = re.findall(patt, markedText)
    sentences = sent_tokenize(text)
    sentenceBoundaries = []
    for sent in sentences:
        sInd = text.index(sent)
        sentenceBoundaries.append((sInd, sInd+len(sent)))
    for i, line in enumerate(lines):
        curOut = {}
        curOut["error type"] = eType
        sentUnderConsideration = sentences[int(line[line.index(". S")+len(". S"):].strip())-1]
        curOut["start"] = text.index(sentUnderConsideration)
        curOut["end"] = curOut["start"] + len(sentUnderConsideration)
        errors.append(curOut)
    return errors
#---------------------------------------------------------------------------
def extractErr(text, markedText, eType="Irrelevant", explanation=""):
    errors = []
    patt = "\[(.*?)\]"
    for span in re.findall(patt, markedText):
        newErr = {
            "error type":eType,
            "explanation":explanation,
        }
        if span not in text:
            logging.warning("Unable to find span {} in {}".format(span, text))
            continue
        startInd = text.index(span)
        endInd = min(len(text), startInd+len(span))
        newErr["start"] = startInd
        newErr["end"] = endInd
        errors.append(newErr)
    return errors
#---------------------------------------------------------------------------
def removeMarkedSpans(markedText):
    patt = "\[(.*?)\]"
    markedText = re.sub("\[[ ]*\]", "", markedText)
    if len(re.findall(patt, markedText)) and markedText != "[]":
        return re.sub(patt, "", markedText)
    return None
#---------------------------------------------------------------------------
def consolidateErrors(feedback, types=["relevance", "factuality"]):
    assert len(types) >= 1
    eTypesUnderConsideration = []
    for t in types:
        eTypesUnderConsideration.extend(visualizeToEtypes[t])

    consolidatedErrs = []
    errSpans = {}
    for err in feedback["errors"]:
        if err["error type"] not in eTypesUnderConsideration:
            consolidatedErrs.append(err.copy())
            continue
        if err["error type"] not in errSpans.keys():
            errSpans[err["error type"]] = []
        errSpans[err["error type"]].append((err["start"], err["end"]))
    for eType in errSpans.keys():
        errSpans[eType].sort()
        i = 0
        while i < len((errSpans[eType])):
            curStart = errSpans[eType][i][0]
            curEnd = errSpans[eType][i][1]
            if i < (len(errSpans[eType])-1):
                #If next error starts after current error ends, nothing to handle
                if errSpans[eType][i+1][0] >= curEnd: 
                    pass 
                else: 
                    #If next error ends before current error ends, skip next error
                    while i < (len(errSpans[eType])-1) and errSpans[eType][i+1][1] <= curEnd:
                        i += 1  
                    if i < (len(errSpans[eType])-1): 
                        errSpans[eType][i+1] = (curEnd, errSpans[eType][i+1][1])
            consolidatedErrs.append({
                "error type": eType,
                "start": curStart,
                "end": curEnd,
            })
            i += 1
    feedback.update({
        "errors": consolidatedErrs,
    })
    return feedback
            
#---------------------------------------------------------------------------
def extractMissingSentences(markedText, eType="Missing-Answer"):
    missingInfo = []
    
    # splitText = markedText.split("Explanation:\n")
    # markedText, explanations = splitText[0].strip(), splitText[1].strip().split("\n")
    # explanations = [".".join(exp.strip().split(".")[1:]) if "." in exp else exp for exp in explanations]
    
    patt = '[0-9]*\\. Passage [0-9]*, sentence [0-9]* \[.*\]'
    lines = re.findall(patt, markedText)

    # assert len(explanations) == len(lines)

    for i, line in enumerate(lines):
        curOut = {}
        eTypeStart = line.index("[")+1
        eTypeEnd = line.index("]")
        eTypeDetermined = line[eTypeStart:eTypeEnd]
        if eTypeDetermined in STR_2_ETYPE.keys():
            curOut["error type"] = STR_2_ETYPE[eTypeDetermined]
        else:
            curOut["error type"] = eType
        curOut["passage_id"]  = int(line[line.index("Passage")+len("Passage"):].split(",")[0])
        curOut["sentence_id"] = [int(line[line.index("sentence")+len("sentence"):line.index("[")].split("\n")[0])]
        # curOut["explanation"] = explanations[i]
        missingInfo.append(curOut)
    return missingInfo
#---------------------------------------------------------------------------
def consolidateMissingSentences(feedback):
    missingSents = {}
    consolidatedMS = []
    for ms in feedback["missing-info"]:
        if ms["passage_id"] not in missingSents.keys():
            missingSents[ms["passage_id"]] = []
        if ms["sentence_id"] not in missingSents[ms["passage_id"]]:
            consolidatedMS.append(ms.copy())
            missingSents[ms["passage_id"]].append(ms["sentence_id"])
    feedback.update({
        "missing-info": consolidatedMS,
    })
    return feedback
#---------------------------------------------------------------------------
def consolidateModelFeedback(modelFeedback, types=["relevance", "factuality", "completeness"]):
    if "relevance" in types or "factuality" in types:
        modelFeedback = consolidateErrors(modelFeedback, types)
    if "completeness" in types:
        modelFeedback = consolidateMissingSentences(modelFeedback)
    return modelFeedback
#---------------------------------------------------------------------------
def printPassages(passages, pIDs=None):
    if pIDs == None: 
        pIDs = np.arange(len(passages))
    printPass = []
    for i in range(len(passages)):
        if i not in pIDs:
            continue
        curPass = ["Passage " + str(i+1) + ": Title (S0) - " + passages[i][0].replace("\n", " ")]
        for j in range(1, len(passages[i])):
            curPass.append("S"+str(j)+". "+passages[i][j].replace("\n", " "))
        printPass.append("\n".join(curPass))
    return "\n\n".join(printPass)
#---------------------------------------------------------------------------
def printSentences(text):
    sentences = sent_tokenize(text)
    out = []
    for sNo, sent in enumerate(sentences):
        out.append("S{}. {}".format(sNo+1, sent.replace("\n", " ")))
    return "\n".join(out)
#---------------------------------------------------------------------------
def _colourSpans(annotationSpans, goldSpans):
    yellow, red, green = [], [], []
    annotationSpans.sort()
    goldSpans.sort()
    for aInd in range(len(annotationSpans)):
        lastGreen = annotationSpans[aInd][0]
        for gInd in range(len(goldSpans)):
            #If the gold span end before/at the start of the current annotated span, there would be no overlap with this gold span
            if goldSpans[gInd][1] <= annotationSpans[aInd][0]:
                continue
            #If the gold span starts after the current annotated span ends, there would be no overlap with this gold span or any successive gold span (gold spans are sorted)
            if goldSpans[gInd][0] > annotationSpans[aInd][1]:
                break
            #If the gold span starts after the last green span on the current annotated span => The span starting from where the last green span ended until where the current gold span starts is to be marked red
            if goldSpans[gInd][0] > lastGreen:
                red.append((lastGreen, goldSpans[gInd][0]))
            green.append((max(goldSpans[gInd][0], lastGreen), min(goldSpans[gInd][1], annotationSpans[aInd][1])))
            lastGreen = min(goldSpans[gInd][1], annotationSpans[aInd][1])
        if lastGreen < annotationSpans[aInd][1]:
            red.append((lastGreen, annotationSpans[aInd][1]))
    for gInd in range(len(goldSpans)):
        lastGreen = goldSpans[gInd][0]
        for greenInd in range(len(green)):
            #If the green span ends before the current gold span starts, there would be no overlap with this green span
            if green[greenInd][1] <= goldSpans[gInd][0]:
                continue 
            #If the green span starts after the current gold span ends, there would be no overlap with this green span or any successive green span (green spans are sorted)
            if green[greenInd][0] > goldSpans[gInd][1]:
                break 
            #If the green span starts after the start of the current gold span, the span starting from where the last green span ended to where this green span starts should be marked in yellow
            if green[greenInd][0] > goldSpans[gInd][0]:
                yellow.append((lastGreen, min(goldSpans[gInd][1], green[greenInd][0])))
            lastGreen = green[greenInd][1]
        if lastGreen < goldSpans[gInd][1]:
            yellow.append((lastGreen, goldSpans[gInd][1]))
    colouredSpans = []
    for y in yellow:
        colouredSpans.append((y[0], y[1], "magenta"))
    for g in green:
        colouredSpans.append((g[0], g[1], "green"))
    for r in red:
        colouredSpans.append((r[0], r[1], "red"))
    colouredSpans.sort()
    return colouredSpans
#---------------------------------------------------------------------------
def _highlight(colour, text):
    if colour == "black":
        return "\033[1;40m" + str(text) + "\033[1;m"
    if colour == "red":
        return "\033[1;41m" + str(text) + "\033[1;m"
    if colour == "green":
        return "\033[1;42m" + str(text) + "\033[1;m"
    if colour == "yellow":
        return "\033[1;43m" + str(text) + "\033[1;m"
    if colour == "blue":
        return "\033[1;44m" + str(text) + "\033[1;m"
    if colour == "magenta":
        return "\033[1;45m" + str(text) + "\033[1;m"
    if colour == "cyan":
        return "\033[1;46m" + str(text) + "\033[1;m"
    if colour == "gray":
        return "\033[1;47m" + str(text) + "\033[1;m"
    return str(text)
#---------------------------------------------------------------------------
def visualizeText(text, annotation, gold, eTypes):
    text = text
    goldSpans = []
    annotationSpans = []
    for gInd in range(len(gold)): 
        if gold[gInd]["error type"] not in eTypes:
            continue
        goldSpans.append((gold[gInd]["start"], gold[gInd]["end"]))
        
    for aInd in range(len(annotation)): 
        if annotation[aInd]["error type"] not in eTypes:
            continue
        annotationSpans.append((annotation[aInd]["start"], annotation[aInd]["end"]))
    colouredSpans = _colourSpans(annotationSpans, goldSpans)
    # print(colouredSpans)
    colouredText = ""
    last = 0 
    for i in range(len(colouredSpans)):
        colouredText += _highlight("white", text[last:colouredSpans[i][0]])
        colouredText += _highlight(colouredSpans[i][-1], text[colouredSpans[i][0]: min(len(text), colouredSpans[i][1])])
        last = min(len(text), colouredSpans[i][1])
    colouredText += _highlight("white", text[last:].strip())
    print(colouredText)
#---------------------------------------------------------------------------
def _computePrecision(prediction, annotationSpansFact, goldSpansRel, wordLevel=False):
    precisionRelevance = 0
    aSum = 0
    for aInd in range(len(annotationSpansFact)):
        curPrecision = 0
        if wordLevel:
            aSum += len(word_tokenize(prediction[annotationSpansFact[aInd][0]:annotationSpansFact[aInd][1]]))
        else: 
            aSum += (annotationSpansFact[aInd][1] - annotationSpansFact[aInd][0])
        for gInd in range(len(goldSpansRel)):
            if goldSpansRel[gInd][1] < annotationSpansFact[aInd][0]:
                continue
            # gold start after annotation end
            if goldSpansRel[gInd][0] > annotationSpansFact[aInd][1]:
                break
            # gold start before annotation end
            else:
                if wordLevel:
                    curPrecision += len(word_tokenize(prediction[max(goldSpansRel[gInd][0], annotationSpansFact[aInd][0]):min(goldSpansRel[gInd][1], annotationSpansFact[aInd][1])]))
                else: 
                    curPrecision += (min(goldSpansRel[gInd][1], annotationSpansFact[aInd][1]) - max(goldSpansRel[gInd][0], annotationSpansFact[aInd][0]))
                if goldSpansRel[gInd][1] > annotationSpansFact[aInd][1]:
                    break
        precisionRelevance += curPrecision
    if aSum:
        precisionRelevance /= aSum
    else:
        if len(goldSpansRel) == 0 and len(annotationSpansFact) == 0:
            precisionRelevance = 1
        else:
            precisionRelevance = 0
    return precisionRelevance
#---------------------------------------------------------------------------
nlp = spacy.load('en_core_web_sm') # Load the English Model
MIN_SUBSENT_WORDS = 5
MIN_SIM_SCORE = 0.85
#---------------------------------------------------------------------------
def _find_approximate_matching_sequence(context, target):
    """ Find some substring in the context which closely matches the target, returning this substring with a score.
        Source: https://github.com/apple/ml-qrecc/blob/main/utils/span_heuristic.py
    """
    if target in context:
        return target, 1.0

    target_length = len(target.split())
    max_sim_val = 0
    max_sim_string = ''
    seq_matcher = SM()
    seq_matcher.set_seq2(target)
    for ngram in ngrams(context.split(), target_length + int(0.05 * target_length)):
        candidate_ngram = ' '.join(ngram)
        seq_matcher.set_seq1(candidate_ngram)
        similarity = seq_matcher.quick_ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = candidate_ngram
        if similarity == 1.0:
            # early exiting
            break

    return max_sim_string, max_sim_val
#---------------------------------------------------------------------------
def get_subsentence_starts(tokens):
    """Get the indices of the tokens that start a subsentence."""
    def _is_tok_end_of_subsent(tok):
        if re.match('[,;!?]', tok[-1]) is not None:
            return True
        return False

    assert len(tokens) > 0
    is_subsent_starts = [True]
    prev_tok = tokens[0]
    prev_subsent_start_idx = 0
    for i, tok in enumerate(tokens[1:]):
        tok_id = i + 1
        if _is_tok_end_of_subsent(prev_tok) and tok_id + MIN_SUBSENT_WORDS < len(tokens):
            if tok_id - prev_subsent_start_idx < MIN_SUBSENT_WORDS:
                if prev_subsent_start_idx > 0:
                    is_subsent_starts += [True]
                    is_subsent_starts[prev_subsent_start_idx] = False
                    prev_subsent_start_idx = tok_id
                else:
                    is_subsent_starts += [False]
            else:
                is_subsent_starts += [True]
                prev_subsent_start_idx = tok_id
        else:
            is_subsent_starts += [False]
        prev_tok = tok
    return [i for i, is_start in enumerate(is_subsent_starts) if is_start]
#---------------------------------------------------------------------------
def find_best_p_sents(text, all_p_sents):
    """ Find the best passage sentences for each subsentence in the LM predicted output. """
    doc = nlp(' '.join(text.strip().split()))
    subsents = []

    for s in doc.sents:
        s_text = s.text
        tokens = s_text.split()
        sent_start_idx = get_subsentence_starts(tokens)
        for i, idx in enumerate(sent_start_idx):
            if i < len(sent_start_idx) - 1:
                subsents += [' '.join(tokens[idx:sent_start_idx[i+1]])]
            else:
                subsents += [' '.join(tokens[idx:])]

    res = []
    for subsent in subsents:
        max_sim = -1
        max_sim_sent = ""
        max_sim_sent_id = (-1, -1)
        for sent_id, p_sent in all_p_sents.items():
            subsent_length = len(subsent.split())
            p_sent_length = len(p_sent.split())
            diff = subsent_length + int(0.05 * subsent_length) - p_sent_length + 1
            if diff > 0:  # add some padding, otherwise the returned sim score is always 0
                _, cur_max_sim = _find_approximate_matching_sequence(p_sent+' #' * diff, subsent)
            else:
                _, cur_max_sim = _find_approximate_matching_sequence(p_sent, subsent)
            if cur_max_sim > max_sim:
                max_sim = cur_max_sim
                max_sim_sent = p_sent
                max_sim_sent_id = sent_id
        res += [(subsent, max_sim_sent, max_sim_sent_id, max_sim)]
    
    return res
#---------------------------------------------------------------------------
def get_coverage_score(pred, all_p_sents, all_info_ids):
    """Compute the percentage of information ids that are covered by the LM predicted output."""
    
    # map LM output to passage sentences
    pred_p_sents = find_best_p_sents(pred, all_p_sents)
    pred_info_ids = set()

    for sent in pred_p_sents:
        subsent, sim_sent, sim_sent_id, sim = sent

        if sim < MIN_SIM_SCORE:
            continue

        info_id = (sim_sent_id[0], sim_sent_id[1])
        pred_info_ids.add(info_id)
    
    score = len(pred_info_ids.intersection(all_info_ids)) * 1.0 / len(all_info_ids)

    return score
#---------------------------------------------------------------------------
def scoreCorrectedPrediction(annotation, gold, passages):
    # read all passage sentences
    all_p_sents = {}
    for i, p in enumerate(passages):
        for j, sent in enumerate(p[1:]):
            all_p_sents[(i+1, j+1)] = ' '.join(sent.split())

    # some sentences only contain minor relevant info
    minor_info_ids = set()
    for m in gold["missing-info"]:
        if m['error type'] == "Missing-Minor-Auxiliary":
            for sid in m["sentence_id"]:
                minor_info_ids.add((m['passage_id'], sid))

    # get all passage sentences with relevant info by mapping each subsent 
    # in human-written corrections to the grounding passage sentences
    # read such sentences as all_info_ids
    correct = gold["corrected-prediction"]
    correct = ' '.join(correct.strip().split())
    correct_p_sents = find_best_p_sents(correct, all_p_sents)
    all_info_ids = set()
    minus_info_ids = set()
    for sent in correct_p_sents:
        subsent, sim_sent, sim_sent_id, sim = sent
        info_id = (sim_sent_id[0], sim_sent_id[1])
        if sim < MIN_SIM_SCORE:
            continue
        all_info_ids.add(info_id)

        # do not count sentences with minor info only
        if info_id in minor_info_ids:
            minus_info_ids.add(info_id)
            if subsent.lower() in annotation["corrected-prediction"].lower():
                minus_info_ids.discard(info_id)

    all_info_ids = all_info_ids.difference(minus_info_ids)

    # skip examples without relevant info (could be because of 
    # no useful grounding passage)
    if len(all_info_ids) == 0:
        return 1

    # match each LM output to all relevant info and calcalate the completion ratio
    pred = annotation["corrected-prediction"]
    score = get_coverage_score(pred, all_p_sents, all_info_ids)

    return score
#---------------------------------------------------------------------------
def evaluateAnnotation(prediction, annotation, gold, passages, order):
    evaluation = {}
    if "R" in order:
        goldSpansRel = []
        annotationSpansRel = []
    if "F" in order:
        goldSpansFact = []
        annotationSpansFact = []

    sentences = sent_tokenize(prediction)
    sentenceBoundaries = []
    for sent in sentences:
        sInd = prediction.index(sent)
        sentenceBoundaries.append((sInd, sInd+len(sent)))
        
    if "R" in order or "F" in order:
        for g in gold["errors"]:
            if "R" in order and g["error type"] in RELEVANCE_CATEGORIES:
                goldSpansRel.append((g["start"], g["end"]))
            elif "F" in order and g["error type"] in FACTUALITY_CATEGORIES:
                for sbInd, sb in enumerate(sentenceBoundaries):
                    if sb[0] <= g["start"] <= sb[1]:
                        if sb not in goldSpansFact:
                            goldSpansFact.append(sb)
                        if sb[0] <= g["end"] <= sb[1]:
                            break
                    if sb[0] <= g["end"] <= sb[1]:
                        if sb not in goldSpansFact:
                            goldSpansFact.append(sb)
                        break
        
        for g in annotation["errors"]:
            if "R" in order and g["error type"] in RELEVANCE_CATEGORIES:
                annotationSpansRel.append((g["start"], g["end"]))
            elif "F" in order and g["error type"] in FACTUALITY_CATEGORIES:
                for sbInd, sb in enumerate(sentenceBoundaries):
                    if sb[0] <= g["start"] <= sb[1]:
                        if sb not in annotationSpansFact:
                            annotationSpansFact.append(sb)
                        if sb[0] <= g["end"] <= sb[1]:
                            break
                    if sb[0] <= g["end"] <= sb[1]:
                        if sb not in annotationSpansFact:
                            annotationSpansFact.append(sb)
                        break
            else: 
                raise RuntimeError("Unexpected error type: {} for order {}!".format(g["error type"], order))

        if "R" in order:
            goldSpansRel.sort()
            annotationSpansRel.sort()

        if "F" in order:
            goldSpansFact.sort()
            annotationSpansFact.sort()

        if "R" in order:
            #RELEVANCE
            precisionRelevance, recallRelevance, f1Relevance = 0, 0, 0

            precisionRelevance = _computePrecision(prediction, annotationSpansRel, goldSpansRel, wordLevel=True)
            recallRelevance = _computePrecision(prediction, goldSpansRel, annotationSpansRel, wordLevel=True)
            if precisionRelevance and recallRelevance:
                f1Relevance = 2/((1/precisionRelevance)+(1/recallRelevance))
            
            evaluation["relevance"] = {
                "precision": precisionRelevance,
                "recall": recallRelevance,
                "f1": f1Relevance,
            }

        if "F" in order:
            #FACTUALITY
            precisionFactuality, recallFactuality, f1Factuality = 0, 0, 0

            precisionFactuality = _computePrecision(prediction, annotationSpansFact, goldSpansFact)
            recallFactuality = _computePrecision(prediction, goldSpansFact, annotationSpansFact)
            if precisionFactuality and recallFactuality:
                f1Factuality = 2/((1/precisionFactuality)+(1/recallFactuality))

            evaluation["factuality"] = {
                "precision": precisionFactuality,
                "recall": recallFactuality,
                "f1": f1Factuality,
            }

    if "C" in order:
        #MISSING INFO
        goldMissingInfo = set()
        annotationMissingInfo = set()
        precisionMissingInfo, recallMissingInfo, f1MissingInfo = 0, 0, 0
        for g in gold["missing-info"]:
            for sId in g["sentence_id"]:
                goldMissingInfo.add(str((g["passage_id"], sId, g["error type"])))
        for g in annotation["missing-info"]:
            for sId in g["sentence_id"]:
                annotationMissingInfo.add(str((g["passage_id"], sId, g["error type"])))
        if len(annotationMissingInfo):
            precisionMissingInfo = len(annotationMissingInfo.intersection(goldMissingInfo))/len(annotationMissingInfo)
        elif len(goldMissingInfo) == 0:
            precisionMissingInfo = 1

        if len(goldMissingInfo):
            recallMissingInfo = len(goldMissingInfo.intersection(annotationMissingInfo))/len(goldMissingInfo)
        elif len(annotationMissingInfo) == 0:
            recallMissingInfo = 1

        if precisionMissingInfo and recallMissingInfo:
            f1MissingInfo = 2/((1/precisionMissingInfo)+(1/recallMissingInfo))
    
        evaluation["missing-info"] = {
            "precision": precisionMissingInfo,
            "recall": recallMissingInfo,
            "f1": f1MissingInfo,
        }

        evaluation["corrected-prediction"]  = {
            "score": scoreCorrectedPrediction(annotation, gold, passages),
        }
    return evaluation
#---------------------------------------------------------------------------
def updateEvaluation(evaluation, update):
    for outerKey in evaluation.keys():
        if outerKey not in update.keys():
            continue
        for innerKey in evaluation[outerKey].keys():
            if innerKey not in update[outerKey].keys():
                continue
            evaluation[outerKey][innerKey] += update[outerKey][innerKey]
    return evaluation
#---------------------------------------------------------------------------
def averageEvaluation(evaluation, numSamples):
    for outerKey in evaluation.keys():
        for innerKey in evaluation[outerKey].keys():
            evaluation[outerKey][innerKey] /= numSamples
    return evaluation
#---------------------------------------------------------------------------
def main(errTrace="main"):
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)
    
    if args.numShots < 0:
        raise ValueError("[{}] numShots has to be non-negative!".format(errTrace, args.numShots))

    logging.info(args)

    if args.numShots == 0:
        logging.warning("[{}] Not providing few shot demonstrations can cause this code to break in several ways.".format(errTrace))
    
    if args.numRoundsR < 1:
        raise ValueError("[{}] numRoundsR has to be at least 1!".format(errTrace))
    
    if args.numRoundsF < 1:
        raise ValueError("[{}] numRoundsF has to be at least 1!".format(errTrace))
    
    if args.numRoundsC < 1:
        raise ValueError("[{}] numRoundsC has to be at least 1!".format(errTrace))
    
    if args.saveFreq < 1:
        raise ValueError("[{}] saveFreq has to be at least 1!".format(errTrace))

    maxRounds = {
        "relevance": args.numRoundsR,
        "factuality": args.numRoundsF,
        "completeness": args.numRoundsC,
    }

    checkIfExists(args.out, isDir=True, createIfNotExists=True)
    systemPromptFilePath = (args.systemPrompt + args.order + "_" + "noCoT" + ".txt")
    if args.CoT:
        systemPromptFilePath = (args.systemPrompt + args.order + "_" + "CoT" + ".txt")
    checkFile(systemPromptFilePath, "txt")
    systemPrompt = "\n".join(readFile(systemPromptFilePath))
    re.sub(r'"', r'\\"', systemPrompt)
    re.sub(r"'", r"\\'", systemPrompt)

    if args.numShots:
        checkFile(args.fewShot, "json")
        fewShotData = readFile(args.fewShot)
        if args.numShots > len(fewShotData):
            raise RuntimeError("[{}] Cannot sample {} few-shot instances from {} instances!".format(errTrace, args.numShots, len(fewShotData)))

        fewShots = np.random.choice(fewShotData, args.numShots, replace=False)

        fewShotsPrompts = []

        for shot in fewShots:
            curPrompts = [
                "Example:",
                PROMPTS["base"].format(
                    question=shot["question"],
                    referenceResponse=shot["gold"],
                    passages=printPassages(shot["passages"]),
                    response=goldResponse,
                )
            ]

            for i, ao in enumerate(ANNOTATION_ORDER[args.order]):
                if ao == "completeness":
                    goldResponse = getMissingSentences(shot["feedback"])
                elif ao == "relevance":
                    goldResponse = markErr(shot["prediction 1"], shot["feedback"], RELEVANCE_CATEGORIES)
                elif ao == "factuality":
                    goldResponse = markSents(shot["prediction 1"], shot["feedback"], FACTUALITY_CATEGORIES)
                else:
                    raise ValueError("[{}] {} is unrecognized!".format(errTrace, ao))
                
                if args.CoT: 
                    raise RuntimeError("[{}] CoT is not supported when adding few shot demonstrations automatically to system prompt!".format(errTrace))
                else: 
                    if ao == "factuality":
                        curPrompts.append(
                            PROMPTS[ao]["noCoT"].format(
                                responseSentences=printSentences(shot["prediction 1"]),
                                annotation="\n"+goldResponse
                            ).strip()
                        )
                    else:
                        curPrompts.append(
                            PROMPTS[ao]["noCoT"].format(
                                annotation="\n"+goldResponse
                            ).strip()
                        )
            curPrompts.append(
                PROMPTS["corrected-prediction"].format(
                    annotation="\n"+shot["feedback"]["corrected-prediction"]
                ).strip()
            )
            fewShotsPrompts.append("\n\n".join(curPrompts))
        systemPrompt += "\n\n" + "\n".join(fewShotsPrompts)
        re.sub(r'"', r'\\"', systemPrompt)
        re.sub(r"'", r"\\'", systemPrompt)

    checkFile(args.data, "json")
    data = readFile(args.data)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    annotations = []
    if args.CoT:
        annotationFilePath = (args.out + "".join(args.data.split("/")[-1].split(".")[:-1]) + "_" + args.order + "_" + str(args.temperature) + "_CoT" + ".json")
    else: 
        annotationFilePath = (args.out + "".join(args.data.split("/")[-1].split(".")[:-1]) + "_" + args.order + "_" + str(args.temperature) + "_noCoT" + ".json")
    if args.append or args.evalOnly or args.visualize:
        if checkFile(annotationFilePath, returnBool=True):
            with open(annotationFilePath, "r") as f:
                annotations = list(json.load(f))
    numRoundsPerAnnotation = None
    if not args.evalOnly and not args.visualize:
        numRoundsPerAnnotation = {ao: 0 for ao in ANNOTATION_ORDER[args.order]}
        numAnnotations = 0
        if args.dataNum != -1:
            if args.dataNum <= 0:
                raise ValueError("[{}] dataNum has to be a positive number!".format(errTrace))
            if len(data) < args.dataNum:
                raise ValueError("[{}] Cannot sample more instances ({}) than available ({})!".format(errTrace, args.dataNum, len(data)))
            data = np.random.choice(data, args.dataNum, replace=False)
            args.dataStart = 0
            args.dataEnd = len(data)
        for instanceIndex in range(max(args.dataStart, 0), min(args.dataEnd, len(data))):
            numAnnotations += 1
            instance = data[instanceIndex]
            logging.info("Instance {} [{} out of {}]".format(instanceIndex, instanceIndex-max(args.dataStart, 0)+1, min(args.dataEnd, len(data))-max(args.dataStart, 0))) 
            curAnnotation = instance.copy()
            messages=[
                {
                "role": "system",
                "content": systemPrompt
                },
            ]

            curPrompts = [
                PROMPTS["base"].format(
                    question=instance["question"],
                    referenceResponse=instance["gold"],
                    passages=printPassages(instance["passages"]),
                    response=instance["prediction 1"],
                )
            ]

            modelFeedback = {
                "errors": [],
                "missing-info": [],
            }

            for i, ao in enumerate(ANNOTATION_ORDER[args.order]):
                logging.info("\tAnnotating {}".format(ao))
                if args.CoT: 
                    if ao == "factuality":
                        curPrompts.append(
                            PROMPTS[ao]["CoT"].format(
                                responseSentences=printSentences(instance["prediction 1"]),
                                annotation=""
                            ).strip()
                        )
                    else:
                        curPrompts.append(
                            PROMPTS[ao]["CoT"].format(
                                annotation=""
                            ).strip()
                        )
                else: 
                    if ao == "factuality":
                        curPrompts.append(
                            PROMPTS[ao]["noCoT"].format(
                                responseSentences=printSentences(instance["prediction 1"]),
                                annotation=""
                            ).strip()
                        )
                    else:
                        curPrompts.append(
                            PROMPTS[ao]["noCoT"].format(
                                annotation=""
                            ).strip()
                        )

                moreRounds = True
                numRounds = 0
                explanations = []
                messages.append(
                    {
                        "role": "user",
                        "content": "\n".join(curPrompts),
                    }
                )
                lastMessage = None
                while moreRounds:
                    logging.info("\t\tRound {}/{}".format(numRounds+1, maxRounds[ao]))
                    response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        frequency_penalty=args.frequency_penalty,
                        presence_penalty=args.presence_penalty,
                        stop=["######"]
                    )

                    response["choices"][0]["message"]["content"] = "\n\n".join(response["choices"][0]["message"]["content"].split("\n\n")[:(1+args.CoT)])

                    lastMessage = dict(response["choices"][0]["message"])

                    #Extract annotations
                    modelResponse = response["choices"][0]["message"]["content"].split("\n")

                    if args.CoT: 
                        explanations.append(modelResponse[-1])
                    if ao == "completeness":
                        errAnnotation = extractMissingSentences(modelResponse[0].strip())
                        modelFeedback["missing-info"].extend(errAnnotation)
                    elif ao in ["relevance"]:
                        eType = "Irrelevant"
                        errAnnotation = extractErr(instance["prediction 1"], modelResponse[0].strip(), eType)
                        modelFeedback["errors"].extend(errAnnotation)
                    elif ao in ["factuality"]:
                        eType = "Unverifiable"
                        errAnnotation = extractUnfactualSentences(instance["prediction 1"], modelResponse[0].strip(), eType)
                        modelFeedback["errors"].extend(errAnnotation)
                    else:
                        raise ValueError("[{}] {} is unrecognized!".format(errTrace, ao))
                    numRounds += 1
                    if numRounds >= maxRounds[ao]:
                        moreRounds = False
                numRoundsPerAnnotation[ao] += numRounds
                curPrompts = []
                #Consolidate annotations across rounds
                consolidatedContent = lastMessage["content"]
                if maxRounds[ao] > 1:
                    modelFeedback = consolidateModelFeedback(modelFeedback, types=[ao])
                    consolidatedContent = "{consolidatedAnnotation}"
                    if ao == "relevance":
                        consolidatedContent = consolidatedContent.format(
                            consolidatedAnnotation=markErr(instance["prediction 1"], modelFeedback, RELEVANCE_CATEGORIES).strip(),
                        )
                    elif ao == "factuality":
                        consolidatedContent = consolidatedContent.format(
                            consolidatedAnnotation=markErr(instance["prediction 1"], modelFeedback, FACTUALITY_CATEGORIES).strip(),
                        )
                    elif ao == "completeness":
                        consolidatedContent = consolidatedContent.format(
                            consolidatedAnnotation=getMissingSentences(modelFeedback).strip(),
                        )
                    else:
                        raise ValueError("[{}] Consolidtaion of messages not supported for {}!".format(errTrace, ao))
                messages.append(
                    {
                        "role": "assistant",
                        "content": consolidatedContent.strip(),
                    }
                )

            ######
            #Corrected model prediction [START]
            logging.info("\tAnnotating corrected-prediction")
            curPrompts = []
            curPrompts.append(
                PROMPTS["corrected-prediction"].format(
                    annotation="",
                ).strip()
            )
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(curPrompts),
                }
            )
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                stop=["######"]
            )

            response["choices"][0]["message"]["content"] = "\n\n".join(response["choices"][0]["message"]["content"].split("\n\n")[:(1)])
            modelResponse = response["choices"][0]["message"]["content"].split("\n")

            modelFeedback["corrected-prediction"] = modelResponse[0].strip()
            messages.append(
                {
                    "role": "assistant",
                    "content": modelResponse[0].strip(),
                }
            )
            #Corrected model prediction [END]
            ######

            curAnnotation["messages"] = messages
            curAnnotation["feedback_human"] = instance["feedback"].copy()
            curAnnotation["feedback"] = modelFeedback
            annotations.append(curAnnotation)
            #Checkpoint: Save annotations
            if (instanceIndex-max(args.dataStart, 0)+1)%args.saveFreq == 0:
                with open(annotationFilePath, "w") as f:
                    json.dump(annotations, f)
                logging.info("~~~Saved annotations~~~")
        #Save annotations
        with open(annotationFilePath, "w") as f:
                    json.dump(annotations, f)
    if args.visualize:
        for i, a in enumerate(annotations):
            print("Visualizing {}/{}".format(i+1, len(annotations)))
            visualizeText(a["prediction 1"], annotation=a["feedback"]["errors"], gold=a["feedback_human"]["errors"], eTypes=visualizeToEtypes[args.visualize])
    else:
        #Evaluation
        evaluation = {}
        if "R" in args.order:
            evaluation["relevance"] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }
        if "F" in args.order:
            evaluation["factuality"] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }
        if "C" in args.order:
            evaluation["missing-info"] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }
        evaluation["corrected-prediction"] = {
            "score": 0,
        }
        for curAnnotation in annotations:
            update = evaluateAnnotation(
                prediction=curAnnotation["prediction 1"],
                annotation=curAnnotation["feedback"], 
                gold=curAnnotation["feedback_human"],
                passages=curAnnotation["passages"],
                order=args.order,
            )
            evaluation = updateEvaluation(evaluation, update)
        evaluation = averageEvaluation(evaluation, len(annotations))
        print("Evaluation")
        print("No. of instances: {}".format(len(annotations)))
        for k in evaluation.keys():
            print(k)
            for kk in evaluation[k].keys():
                print("\t{}: {:0.2f}".format(kk, evaluation[k][kk]))
        if numRoundsPerAnnotation != None:
            print("Average no. of rounds per annotation:")
            for ao in numRoundsPerAnnotation.keys():
                print("\t{}: {:0.2f}".format(ao, numRoundsPerAnnotation[ao]/numAnnotations))
#---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# containsAll = []
# containsSum = []
# containsRel = []
# containsFact = []
# for i in range(len(data)):
#     rel = [0]*len(RELEVANCE_CATEGORIES)
#     fact = [0]*len(FACTUALITY_CATEGORIES)
#     for err in data[i]["feedback"]["errors"]:
#         if err["error type"] in RELEVANCE_CATEGORIES:
#             rel[RELEVANCE_CATEGORIES.index(err["error type"])] = 1
#         elif err["error type"] in FACTUALITY_CATEGORIES:
#             fact[FACTUALITY_CATEGORIES.index(err["error type"])] = 1
#         else:
#             raise RuntimeError(err["error type"])
#     containsRel.append(sum(rel))
#     containsFact.append(sum(fact))
#     containsSum.append(sum(rel) + sum(fact))
#     if (sum(rel) + sum(fact)) == (len(RELEVANCE_CATEGORIES) + len(FACTUALITY_CATEGORIES)):
#         containsAll.append(i)

def prompt(instance):
    pmt = []
    for i, passage in enumerate(instance["passages"]): 
        pmt.append("Passage {}:\n{}\n".format(i, passage["content"]))
    pmt.append("Question: {}\nGive the rationale before answering.".format(instance["question"]))
    return "\n".join(pmt)
print(prompt(data[1000]))
import argparse
import logging
import regex as re
import json 

RELEVANCE_CATEGORIES = ["Irrelevant", "Incoherent", "Redundant"]
FACTUALITY_CATEGORIES = ["Unverifiable", "Wrong-Grounding"]

#---------------------------------------------------------------------------------------------
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
#---------------------------------------------------------------------------------------------
def extractErr(text, markedText, eType="Irrelevant", explanation=""):
    errors = []
    patt = "\[(.*?)\]"
    for span in re.findall(patt, markedText):
        newErr = {
            "error type":eType,
            "explanation":explanation
        }
        startInd = text.index(span)
        endInd = startInd+len(span)
        newErr["start"] = startInd
        newErr["end"] = endInd
        errors.append(newErr)
    return errors
#---------------------------------------------------------------------------------------------
def extractMissingSentences(instance):
    sents = []
    out = []
    for i in range(len(instance["feedback"]["missing-info"])):
        eType = instance["feedback"]["missing-info"][i]["error type"]
        pId = instance["feedback"]["missing-info"][i]["passage_id"]
        pInd = pId-1
        for sId in instance["feedback"]["missing-info"][i]["sentence_id"]:
            sInd = sId-1
            sents.append((eType, instance["passages"][pInd][sInd]))
            out.append("{}. Passage {}, sentence {}".format(len(out)+1, pId, sId))
    if len(out) == 0:
        out = ["None"]
    return sents, "\n".join(out)
#---------------------------------------------------------------------------------------------
def printPassages(passages):
    printPass = []
    for i in range(len(passages)):
        curPass = ["Passage " + str(i+1) + ": Title (S0) - " + passages[i][0].replace("\n", " ")]
        for j in range(1, len(passages[i])):
            curPass.append("S"+str(j)+". "+passages[i][j].replace("\n", " "))
        printPass.append("\n".join(curPass))
    return "\n\n".join(printPass)
#---------------------------------------------------------------------------------------------
def main():
    trainPath = "./train_feedback.json"
    with open(trainPath, "r") as f:
        data = list(json.load(f))

    prompts = ["""Input Question:
{question}

{passages}

Reference response:
{referenceResponse}

Model Prediction:
{response}

Mark the irrelevant and incoherent spans in the model prediction by enclosing them within square brackets ([]) and then explain your reasoning.

Annotated Model Prediction:
""",
"""Now,  mark the spans in the model prediction that contain inconsistent or unverifiable facts by enclosing them within square brackets ([]) and then explain your reasoning.

Annotated Model Prediction:""",
"""Finally, mark missing information in the model prediction by listing out the passage and sentence numbers containing the missing information. Also explain your reasoning.

Missing Info:"""
]


    for i, d in enumerate(data):
        if i != 10:
            continue
        curPrompt = prompts[0].format(
            question=d["question"],
            passages=printPassages(d["passages"]),
            referenceResponse=d["gold"],
            response=d["prediction 1"],
        )
        goldResponse = markErr(d["prediction 1"], d["feedback"], RELEVANCE_CATEGORIES)
        print(curPrompt)
        print(goldResponse)

        curPrompt = prompts[1]
        goldResponse = markErr(d["prediction 1"], d["feedback"], FACTUALITY_CATEGORIES)
        print(curPrompt)
        print(goldResponse)

        curPrompt = prompts[2]
        _, goldResponse = extractMissingSentences(d)
        print(curPrompt)
        print(goldResponse)

        print(d["feedback"])

        exit(0)
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

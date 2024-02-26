#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:00:00
#SBATCH --mem=60GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/fgrlhfEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate fgrlhfEnv

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export OPENAI_API_KEY="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"

ORDER="RFC"
MODEL="gpt-4"
NUMSHOTS=0
TEMPERATURE=0
START=0
END=1
EVAL_ONLY=false
COT=false
APPEND=false
RANDOM_SAMPLE=-1
while getopts 'acem:n:o:r:s:l:t:' opt; do 
  case "$opt" in 
    a) APPEND=true  ;;
    c) COT=true ;;
    e) EVAL_ONLY=true ;;
    m) MODEL="$OPTARG"  ;;
    n) NUMSHOTS="$OPTARG" ;;
    o) ORDER="$OPTARG" ;;
    r) RANDOM_SAMPLE="$OPTARG" ;;
    s) START="$OPTARG"  ;;
    l) END="$OPTARG" ;;
    t) TEMPERATURE="$OPTARG" ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

if [ "$EVAL_ONLY" = true ] ; then
  if [ "$COT" = true ] ; then
    python3 annotate.py -info -dataNum $RANDOM_SAMPLE -CoT -evalOnly -order $ORDER -model $MODEL -temperature $TEMPERATURE -numShots $NUMSHOTS
  else 
    python3 annotate.py -info -dataNum $RANDOM_SAMPLE -evalOnly -order $ORDER -model $MODEL -temperature $TEMPERATURE -numShots $NUMSHOTS
  fi ;
else 
  if [ "$COT" = true ] ; then
    if [ "$APPEND" = true ] ; then
      python3 annotate.py -info -dataNum $RANDOM_SAMPLE -append -CoT -order $ORDER -model $MODEL -temperature $TEMPERATURE -dataStart $START -dataEnd $END -numShots $NUMSHOTS
    else
      python3 annotate.py -info -dataNum $RANDOM_SAMPLE -CoT -order $ORDER -model $MODEL -temperature $TEMPERATURE -dataStart $START -dataEnd $END -numShots $NUMSHOTS
    fi ;
  else 
    if [ "$APPEND" = true ] ; then
      python3 annotate.py -info -dataNum $RANDOM_SAMPLE -append -order $ORDER -model $MODEL -temperature $TEMPERATURE -dataStart $START -dataEnd $END -numShots $NUMSHOTS
    else
      python3 annotate.py -info -dataNum $RANDOM_SAMPLE -order $ORDER -model $MODEL -temperature $TEMPERATURE -dataStart $START -dataEnd $END -numShots $NUMSHOTS
    fi ;
  fi ;
fi ;

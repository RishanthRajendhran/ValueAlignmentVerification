#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/fgrlhfEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate fgrlhfEnv

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/huggingface_cache"

python3 removeRedundancy.py \
    -info \
    -pad_to_max_length \
    -data ./data/dev_feedback.json \
    -loadFeats

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -data ./proData/F-ERR_sentence/dev.json \
#     -factuality
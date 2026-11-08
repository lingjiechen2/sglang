#!/usr/bin/env bash
#SBATCH --account=arch
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=sglang-perexp-expert-test
#SBATCH --output=/mnt/weka/shrd/k2m/lingjie.chen/mova-lora-tied-tests-20261031/audit/logs/%x-%j.out
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
R=/mnt/weka/shrd/k2m/lingjie.chen/sglang-uno-recycle-perexp
cd $R/test/registered/unit/spec
PY=/mnt/weka/shrd/k2m/lingjie.chen/envs/sglang-official-dflash2/bin/python
export PYTHONPATH=$R/python
$PY -m unittest -v test_uno_mova_perexp_expert_lora test_uno_mova_expert_lora 2>&1 | tail -40

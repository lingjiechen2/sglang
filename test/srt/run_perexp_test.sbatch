#!/usr/bin/env bash
#SBATCH --account=arch
#SBATCH --partition=cpuonly
#SBATCH --qos=lowprio
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=sglang-perexp-test
#SBATCH --output=/mnt/weka/shrd/k2m/lingjie.chen/mova-lora-tied-tests-20261031/audit/logs/%x-%j.out
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
/mnt/weka/shrd/k2m/lingjie.chen/envs/sglang-official-dflash2/bin/python -I /mnt/weka/shrd/k2m/lingjie.chen/sglang-uno-recycle-perexp/test/srt/test_uno_mova_perexp_value_lora.py

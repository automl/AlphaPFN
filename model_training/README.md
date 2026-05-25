# Model training

This directory retrains the four v1 checkpoints that ship with the
package — `ppd` (the base model) and the three direct heads
`pes`, `mes`, `jes`.

## 1. Install

```bash
pip install -e ".[training]"
```

Training needs CUDA, PyTorch ≥ 2.0, and ~64 GB of host RAM.
Multi-GPU runs use `torchrun` (single-node only).

## 2. Get the training data

Each of the four corpora is around **210 GB compressed** and expands
to **~258 GB on disk**. They are published at:

```
https://ml.informatik.uni-freiburg.de/research-artifacts/rakotoah/alpha_pfn/
    alpha_pfn_data_v1_ppd.tar.xz
    alpha_pfn_data_v1_pes.tar.xz
    alpha_pfn_data_v1_mes.tar.xz
    alpha_pfn_data_v1_jes.tar.xz
```

You do not download them manually. The training script fetches the
corpus it needs on first use, verifies its SHA-256 against the
published sidecar, and extracts it into a cache directory of your
choice. Point that cache at a workspace filesystem with plenty of
room — **never `$HOME`**, which is small on most clusters:

```bash
export ALPHAPFN_DATA_CACHE_DIR=/scratch/$USER/alphapfn_data
```

After the first run for a given corpus, the extracted data lives at
`$ALPHAPFN_DATA_CACHE_DIR/v1/<corpus>/` and is reused for every
subsequent run.

## 3. Train

On a slurm cluster:

```bash
# PPD base model — 4×H200, ~24 h
bash model_training/slurm/slurm_train_ppd.sh

# PES + MES + JES direct heads — 4×L40s each, submitted in parallel
bash model_training/slurm/slurm_train_alpha_all.sh
```

Hyperparameters default to the v1 recipe; pass `--help` to see the
overrides. Different `--seed` values produce statistically equivalent
(not bit-identical) checkpoints — seed is a parallel-training
convention, not part of the data provenance.

## 4. Export to the inference format

The training loop saves `final_models/<name>.pt`. To make a checkpoint
loadable by `AlphaPFN.from_pretrained(...)`, convert it to the
`config.json` + `weights.safetensors` pair:

```bash
python model_training/export_checkpoints.py \
    --source final_models/ --destination checkpoints/v1/
```


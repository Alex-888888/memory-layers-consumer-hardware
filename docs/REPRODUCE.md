# Reproduction guide

## Environment

The original runs used:

- Windows 11 + **WSL2 Ubuntu 24.04**
- **ROCm 6.4.2**, AMD **RX 7900 XTX (24 GB)**
- **PyTorch 2.5.1+rocm6.2**
- **Qwen2.5-7B-Instruct** (downloaded from Hugging Face on first run, ~15 GB cache)

Nothing here is ROCm-specific in principle; a 24 GB CUDA GPU should work with the matching PyTorch build. Notes on the ROCm/WSL2 setup (library symlinks, `LD_LIBRARY_PATH=/opt/rocm/lib`) are environment-specific and out of scope here.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Generate data

```bash
cd src/data
# memorisation target only:
python make_synthetic.py --n 500 --out ../../data/synthetic_500.jsonl --seed 7
# full mixed corpus (synthetic + public-fact anchor + fluency + instruction):
python make_corpus.py --synth 5000 --out ../../data/corpus.jsonl
```

A tiny deterministic sample (`data/synthetic_sample.jsonl`) is included for a quick smoke test.

## 2. Validate the mechanism (Phase C / stages)

```bash
cd src/stages
python etage2_productkey.py     # product-key non-regression vs exhaustive (KL = 0)
python etage3_memoryplus.py     # Memory+ (gating, shared pool, qk-norm), UUID overfit
python etage4a_qwen_inject.py   # injection into Qwen 6/14/22, frozen backbone (smoke)
```

## 3. Warm-up training (the recipe)

```bash
cd src
python warmup_train.py \
  --corpus jsonl --corpus_path ../data/corpus.jsonl \
  --pool_size 50000 --seq 128 --grad_ckpt 0 --batch 4 \
  --answer_only 1 --pool_optim dense --pack 0 \
  --steps 12000 --warmup_steps 100 --lr 1e-3 \
  --ckpt_every 2000 --ckpt_dir ./checkpoints
```

Key flags (the recipe): `--answer_only 1` (loss on answer tokens only), `--pack 0` (one sequence per fact), `--pool_optim dense` (dense AdamW on the pool). The memory is injected MLP-ADD at layers 6/14/22 with the backbone frozen.

A fast self-contained proof on a few hundred facts:

```bash
python microfit_centered.py --center 0 --nfacts 500 --steps 2500 --eval_n 300
```

## 4. Evaluate recall

```bash
cd src
MLCH_EVAL_CORPUS=../data/corpus.jsonl python eval_factual.py
```

The eval extracts the (entity, value) pairs from the corpus, then reports recall of the **baseline** (memory disabled = frozen model, ~0 % on synthetic) vs the **trained** memory, plus preservation of a set of native known facts.

## Notes

- Long runs benefit from frequent checkpoints and a watchdog that detects a **frozen log** (not just a dead process), because some GPU stalls leave the process alive.
- The pool optimizer choice matters: a sparse/offloaded optimizer can leave the pool near initialisation (see `docs/DIAGNOSTIC.md`); use dense AdamW for pools that fit in VRAM.

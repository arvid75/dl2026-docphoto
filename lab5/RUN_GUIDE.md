# Lab 5 — run guide (GPU machine)

Everything that needs code is already done:

- **TASK 1** — `configs/dropout03.yaml` (one change vs `configs/reference.yaml`: `dropout: 0.0 → 0.3`).
- **TASK 2** — `scripts/oof.py` (out-of-fold RMSE from the four `*_val.csv` files).
- `EXPERIMENTS.md` row 1 is written (fold numbers to be filled after training).

The data lives under `data/raw/` (git-ignored): `train.csv` + `train/` (1841),
`test/` (1628), `sample_submission.csv`. The resized caches under
`data/work/reference/` were built with `python -m src.prepare`.

> On a fresh clone the `data/` folder is empty (git-ignored). Copy the data in
> first (see **Step 0**), then run `prepare`.

---

## Step 0 — data (only on a fresh clone)

Put the competition data into `data/raw/` so it looks like:

```
data/raw/train.csv
data/raw/train/            1841 .jpg
data/raw/test/             1628 .jpg   (Kaggle Data tab → Download All)
data/raw/sample_submission.csv
```

Then build the resized caches (once, ~1 min on a GPU box):

```bash
docker compose build
docker compose run --rm dev python -m src.prepare
# must report: train 1841 + test 1628 resized into data/work/reference
```

## Step 1 — train the four folds (TASK 1)

`--fold k` trains on the other three strict units and validates on fold `k`.
Each fold writes a checkpoint `.pt` and a `*_val.csv`.

```bash
docker compose run --rm dev bash -c 'for k in 0 1 2 3; do \
  python -m src.train --fold $k --config configs/dropout03.yaml --out models/dropout03; done'
```

GPU: ~8 min total. (CPU at 768 px is ~30 min; on CPU also set `image_size: 512`
and compare against the reference's 512 px number, 0.207.)

**Also run the reference**, so you have a baseline to compare the change against:

```bash
docker compose run --rm dev bash -c 'for k in 0 1 2 3; do \
  python -m src.train --fold $k --config configs/reference.yaml --out models/reference; done'
```

## Step 2 — out-of-fold RMSE (TASK 2)

```bash
docker compose run --rm dev python scripts/oof.py models/reference
docker compose run --rm dev python scripts/oof.py models/dropout03
```

Each prints the four per-fold RMSEs and the pooled out-of-fold RMSE. Put both
numbers into `EXPERIMENTS.md`:

- row 0 (reference): the four folds + pooled (~0.194 expected)
- row 1 (dropout03): the four folds + pooled, and mark **kept? yes** if its
  pooled RMSE is lower than the reference's, else **no**.

## Step 3 — predict, check, upload (TASK 3)

Average the four fold checkpoints into one submission (no 5th training run):

```bash
docker compose run --rm dev bash -c \
  'python -m src.predict models/dropout03/*_fold?_*.pt --out submissions/dropout03.csv'
docker compose run --rm dev python scripts/check_submission.py submissions/dropout03.csv
```

`check_submission.py` must pass (1628 rows, header `image_id,usability`, values
in [0,1], not all equal). Then on the competition page → **Submit Prediction**,
upload `submissions/dropout03.csv`; in the description put the change and its
out-of-fold RMSE.

## Step 4 — commit (data/models/submissions stay out of git)

```bash
git add configs/dropout03.yaml scripts/oof.py EXPERIMENTS.md RUN_GUIDE.md
git status          # nothing from data/, models/, submissions/, .env
git commit -m "lab 5: dropout03 config, oof.py, experiment log"
git push
```

## Notes

- **Team name on Kaggle = your Neptun code (`YZG13W`)** — required to match
  submissions to you.
- 5 uploads/day; pick the final 2 by out-of-fold RMSE, not by the public score.
- If no GPU: the same commands run on CPU (slower); or use a Kaggle notebook
  (free GPU) — recipe in the reference `README.md`.
- Native (no Docker) also works: a venv with the packages in `requirements.txt`
  (plus torch/torchvision CPU), then run the same `python -m ...` commands with
  `PYTHONPATH=.` and `WANDB_MODE=offline`.

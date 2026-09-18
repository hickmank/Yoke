LodeRunner Training - DDP - Chicoma
===================================

An example setup training LodeRunner using PyTorch `DistributedDataParallel`
(DDP) on the `lsc240420` layered-shaped-charge dataset. A single timestep of the
per-material density and velocity fields is input and a single (offset) timestep
is predicted. The training system works within limitations but seems more stable
than `lightning.fabric`.

Files
-----

- `train_LodeRunner_ddp.py` — DDP training script for LodeRunner.
- `training_input.tmpl` — single input template. The `<KEY>` tokens are filled
  per study row; the `# <<optional:CONTINUATION>>` block adds `--continuation`
  and `--checkpoint` only on epoch continuation.
- `training_slurm.tmpl` — complete SLURM submission script (Venado GPU partition).
- `eval_LodeRunner.py` — harness-local evaluator: loads a saved checkpoint,
  derives the field ordering from the model's `default_vars`, runs one
  deterministic test pass, and writes per-sample records plus a sidecar
  `*.metadata.json`.
- `evaluation_input.tmpl` / `evaluation_slurm.tmpl` — optional evaluation
  templates. Late-bound tokens `<CHECKPOINT>`, `<INPUTFILE>`, and `<STEM>` are
  filled per checkpoint; the rest come from the CSV row at study creation.
- `cp_files.txt` — files copied into each `study_###` run directory (training
  and evaluation scripts).
- `ddp_paper_study.csv` — hyperparameters for the LodeRunner-18channel paper runs.

Evaluation
----------

Training is a thin wrapper around `yoke.harnesses.trainer.HarnessTrainer` with
`evaluate_after_training=True`, so a **finished** study automatically submits a
separate one-GPU evaluation job for its final checkpoint. Evaluation is a
distinct job (not a phase of training): it does not train, checkpoint, or need
DDP.

To evaluate any other checkpoint — an earlier epoch, a re-run, or a study that
predates this feature — use the manual CLI from this directory:

```bash
yoke-evaluate-study --studyIDX 5 \
    --checkpoint runs/study_005/study005_modelState_epoch0100.pth
```

Add `--dryrun` to render the checkpoint-specific `*.input`/`*.slurm` files and
print the `sbatch` command without submitting. For a study created before the
evaluation feature (no evaluation templates in its `study_###` directory), the
CLI re-renders them on demand from `ddp_paper_study.csv` (pass `--csv`) and
copies `eval_LodeRunner.py` in.

Artifact names derive from the checkpoint *stem*, so an ordinary checkpoint and
its EMA companion (`..._ema.pth`) never collide. To point the automatic
evaluation at an EMA companion, pass `evaluate_checkpoint="ema"` to
`HarnessTrainer` (requires an EMA-enabled harness).

Study
-----

`ddp_paper_study.csv` sweeps LodeRunner model size and temporal offset. Each row
varies the embedding dimension (`EMBED_DIM`), Swin block structure
(`B0`–`B3`), the max time-index offset between input/output (`MAX_TIME_OFFSET`),
node/GPU counts (`KNODES`, `NGPUS`), and batch sizing. The active rows train the
"huge" (`EMBED_DIM=352`) and "giant" (`EMBED_DIM=512`) configurations across 10
nodes; smaller/commented rows are kept for reference.

Platform notes
--------------

1. On Venado, beyond 4 nodes communication conflicts arise intermittently, e.g.:

   ```
   RuntimeError: CUDA error: uncorrectable ECC error encountered
   ```

2. On Venado, with 8 `lsc240420` fields each GPU can only fit 5 samples at a time.
3. On Chicoma, the Giant-size LodeRunner will not fit with DDP training.
4. On Chicoma, the Big-size LodeRunner handles per-GPU batch sizes of 10.

Launch
------

From within this directory, with the Yoke environment active:

```bash
yoke-start-study --csv ddp_paper_study.csv --submissionType slurm
```

Add `--dryrun` to render the study directories and print the `sbatch` commands
without submitting any jobs.

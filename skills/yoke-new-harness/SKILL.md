---
name: yoke-new-harness
description: Interview the user and create a new Yoke training harness under applications/harnesses/ (HarnessTrainer train script, training_input.tmpl, submission template, cp_files.txt, study CSV, README), checking the dataset's batch contract against the epoch function and verifying with yoke-start-study --dryrun. Orchestrates yoke-new-dataset, yoke-new-training-step, and yoke-slurm-template when needed. Use when the user wants to train a Yoke model on a dataset, or set up a new training study or harness.
---

# Create a Yoke Training Harness

You guide the user from "train model X on dataset Y" to a harness directory
that `yoke-start-study` can launch. You also coordinate the other Yoke
skills: a new dataset (`yoke-new-dataset`), a new training step
(`yoke-new-training-step`), and a SLURM template (`yoke-slurm-template`).
Run them when needed, then come back here.

**This skill never submits jobs or trains.** Verification stops at
`yoke-start-study --dryrun` and argument parsing.

Read first:

- `skills/shared/interview_conventions.md` (follow it throughout).
- `AGENTS.md` §4–5a.
- `skills/yoke-new-harness/reference/harness_checklist.md`.
- Any cards from earlier skills (in their `dev_plans/` notes files).

Load when the procedure says to:

- `skills/yoke-new-harness/reference/model_catalog.md`
- `skills/shared/batch_contracts.md`
- `skills/yoke-new-harness/reference/study_csv_and_templates.md`
- `skills/shared/cards.md`

## Procedure

### 1. Orient

- Ask which Python environment has Yoke installed.
- Ask, in a sentence, what the user wants to train: model, data, and goal.
- List `applications/harnesses/`. Propose the **closest existing harness** to
  start from and say why. Copying its structure is much safer than starting
  blank.
- Ask for the harness directory name (existing names use a machine prefix
  such as `ch_` or `vt_`).
- Create the design notes file `dev_plans/<harness_name>_harness.md`.

### 2. Dataset (round 1)

- Which dataset class: from a Dataset card, or an existing class in
  `src/yoke/datasets/`. If none exists, offer `yoke-new-dataset` now and
  resume here afterwards.
- Which constructor arguments become CLI flags (existing ones from
  `yoke.helpers.cli.add_filepath_args`, or new harness flags) and which are
  fixed.
- Train and validation filelists, and the data and filelist paths **on the
  target machine** (these go in `training_input.tmpl`).
- Mixing several datasets (`yoke.datasets.mixed_dataset.MixedDataSet`), with
  weights as CLI flags or CSV columns?

### 3. Model (round 2)

Use `model_catalog.md`:

- Which model. Check that it fits the data: input and output kinds, image
  size against patch and window constraints, `num_input_frames`, and
  `default_vars`.
- Which architecture arguments are fixed in `make_model_args`, exposed as
  flags, or swept in the CSV.
- Fresh training, or starting from a pretrained checkpoint (needs a
  bespoke `model_builder`)?

### 4. Training recipe (round 3)

- **Check the contract.** Look up the dataset's return tuple in
  `batch_contracts.md` and name the epoch function it implies. If the tuple,
  the model call, or the loss shape do not fit any epoch function, **stop**
  and offer `yoke-new-training-step`. Resume with its Step card.
- Loss (`default_mse_loss` vs. `LpLoss` vs. other) and whether it matches the
  datastep's required loss shape.
- Optimizer (AdamW defaults), LR scheduler (cosine with warmup, step, none),
  EMA, grad clip, and `epoch_kwargs` (`channel_map`, `dataset` tag, ...).
- `--batch_size`, `--train_batches`, `--val_batches`, `--TRAIN_PER_VAL`,
  `--total_epochs`, `--cycle_epochs`, `--num_workers`.
- Post-training evaluation: yes or no. Yes needs evaluation templates and
  an evaluator script. Base them on `applications/harnesses/ch_DDP_loderunner/`.

### 5. Submission (round 4)

- `slurm` or `shell`.
- SLURM: reuse the template of an existing harness on the same machine, if
  the user confirms it is current, or run `yoke-slurm-template` (on the
  cluster) and resume with its Slurm card.
- Shell: a `training_shell.tmpl` that runs `python <train_script>
  @<INPUTFILE>`, plus `--submissionType` / `shell` in
  `training_input.tmpl` so continuation resubmits correctly.

### 6. Study design (round 5)

Read `study_csv_and_templates.md`, then ask:

- Which values vary between studies (CSV `<KEY>` columns), and which are
  fixed in `training_input.tmpl`.
- The rows: start with a small **smoke-test CSV** (2–3 rows, few batches),
  then the production CSV.
- Required columns: `studyIDX`, `train_script`, and whatever the submission
  template needs (`KNODES`, `NGPUS`, `allocation`).

### 7. Design and approval

In the notes file, write:

- the file list for the harness dir and what each holds;
- the parser builders and new flags, `make_model_args`, the dataset,
  optimizer, scheduler, and loss builders, the `HarnessTrainer(...)` call,
  `epoch_kwargs`, and hooks;
- the full `training_input.tmpl` and the CSV header with rows;
- the contract check result (dataset tuple, then datastep, then epoch
  function);
- the verification steps (step 9).

Show it and **wait for approval**.

### 8. Implement

Write the files in `applications/harnesses/<name>/`, following
`harness_checklist.md`: the train script, `training_input.tmpl` (ending with
the `# <<optional:CONTINUATION>>` block), the submission template,
`cp_files.txt`, the CSV(s), and `README.md`. Put any reusable logic in
`src/yoke` (with tests), not in the script.

### 9. Verify (no jobs, no training)

Run from inside the harness directory with the user's Python environment,
and render into a **temporary** run directory so no existing `runs/` is
touched:

1. Lint the script: `ruff check`, `ruff check --preview`,
   `ruff format --check --diff`.
2. Render:

   ```bash
   yoke-start-study --csv <smoke.csv> --rundir <tmpdir> \
       --cpFile cp_files.txt --submissionType <type> --dryrun
   ```

   Expect a `[DRY RUN] Would execute: ...` line per row.
3. Look for unfilled tokens in the first-launch files. None may remain:
   `grep -nE "<[A-Za-z_]+>" <tmpdir>/study_*/study*_START.*`.
   (The study's `training_input.tmpl` and submission `.tmpl` copies keep
   `<epochIDX>`, `<INPUTFILE>`, and `<CHECKPOINT>` on purpose.)
4. Parse each rendered `@`-file with the script's own parser. Importing the
   script only builds `parser`, because everything else is under
   `__main__`:

   ```bash
   cd <tmpdir>/study_001 && <python> -c "
   import importlib.util
   spec = importlib.util.spec_from_file_location('h', '<train_script>')
   m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
   print(m.parser.parse_args(['@study001_START.input']))"
   ```

5. Cross-check keys: every CSV column appears as `<COLUMN>` in some template
   (or is intentionally unused), and every `<KEY>` in the templates is a CSV
   column or a late-bound token (`studyIDX`, `epochIDX`, `INPUTFILE`,
   `CHECKPOINT`, `STEM`).
6. Check the model and contract with no data or GPU, if cheap: build the
   model from `make_model_args(parsed_args)` on CPU and count its
   parameters. Ask first if the model is large.
7. Delete the temporary run directory afterwards (it holds only rendered
   files).

Report the actual results. Say what was **not** verified: data paths on the
cluster, the SLURM environment, multi-GPU behaviour, and convergence.

### 10. Summarize

List the files created, the checks and their results, open questions, and
the exact command to launch the study for real, run from the harness
directory:

```bash
yoke-start-study --csv <study.csv> --rundir ./runs --cpFile cp_files.txt \
    --submissionType <type>
```

The user launches it. You do not.

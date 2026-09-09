# Dev Plan: Convert `TandVplot.py` into an installed CLI (`yoke-plot-loss`)

## 1. Goal

Turn the ad-hoc script `applications/evaluation/TandVplot.py` into a reusable,
installable console-script that ships with Yoke — modeled on `yoke-start-study`.

The plan follows Yoke conventions from `AGENTS.md`:
- **Reusable logic lives in `src/yoke`**; the CLI entry point is thin.
- Google docstrings, full type annotations, `<= 89` col lines, double quotes.
- Tests mirror the `src` layout under `tests/`.
- Must pass `pytest -Werror`, `ruff check`, and `ruff format --check`.

## 2. Current state (what exists today)

- `applications/evaluation/TandVplot.py` (276 lines) is a top-level script that:
  - Builds an `argparse` parser at module scope (executes on import).
  - Globs `training_study###_epoch*.csv` and
    `validation_study###_epoch*.csv` under `<basedir>/study_###/`.
  - Reads each CSV with pandas (`sep=", "`, columns `Epoch, Batch, Loss`).
  - Reshapes losses into `Nsamps_per_*_pt` blocks, computes
    `[0.025, 0.5, 0.975]` quantiles, and plots training (blue) + validation
    (red) median +/- quantile bands vs. a running "Evaluation Index".
  - Optionally scatters raw losses, sets a y-limit, drops the last in-progress
    training CSV, and saves or shows the figure.
- The script is **not** importable/testable cleanly (all logic runs at import),
  has no docstrings/annotations on functions (there are no functions), uses bare
  `print` for warnings, and mutates global matplotlib rc state at import.

## 3. Target design

### 3.1 New module: `src/yoke/plots/loss_curves.py`

Create a `plots` subpackage (new `src/yoke/plots/__init__.py` with a module
docstring). Put the reusable, testable functions here. Proposed API:

- `find_record_csvs(basedir, studyIDX, kind, inprogress=False) -> tuple[list[str], list[int]]`
  - Globs training/validation CSVs for one study, returns sorted file list and
    parsed epoch list. `kind` in `{"training", "validation"}`.
  - Encapsulates the current glob patterns and the `inprogress` pop-last logic
    (guarding against empty lists — current code can `IndexError`).
- `load_record_csv(path) -> pandas.DataFrame`
  - Reads a single record CSV (`Epoch, Batch, Loss`) with the existing
    `sep=", ", engine="python"` settings.
- `compute_quantile_bands(losses, nsamps_per_pt, quantiles=(0.025, 0.5, 0.975)) -> tuple[np.ndarray, np.ndarray]`
  - Reshapes into `(nsamps_per_pt, -1)`, trimming the remainder with a
    `logging`/`warnings` message instead of `print`; returns positions + bands.
- `plot_loss_curves(basedir, studyIDX, *, nsamps_per_trn_pt, nsamps_per_val_pt,
  scatter, ylim, inprogress) -> matplotlib.figure.Figure`
  - Orchestrates the above and returns a `Figure` (does NOT save/show — keeps it
    testable and side-effect free). Matplotlib rc setup moves into this function
    (or a small private helper) rather than executing at import time.
- `save_or_show_figure(fig, studyIDX, savefig, savedir) -> str | None`
  - Handles the save-vs-show branch and directory creation; returns the saved
    path (or `None`).

Rationale: separating "compute", "plot", and "output" makes the numeric logic
unit-testable without a display backend, and lets the CLI stay thin.

### 3.2 New CLI arg builder in `src/yoke/helpers/cli.py`

Add `add_plot_loss_args(parser) -> argparse.ArgumentParser` mirroring the
existing `add_*_args` builders. It registers the current flags:

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--basedir` | str | `./runs` | Change default from `./study_directory` to match harness `--rundir` default. |
| `--IDX` / `-I` | int | `0` | Study index. |
| `--Nsamps_per_trn_pt` / `-Nt` | int | `2012` | |
| `--Nsamps_per_val_pt` / `-Nv` | int | `250` | |
| `--scatter` / `-s` | flag | off | |
| `--ylim` / `-Y` | float | `1.0` | |
| `--inprogress` / `-P` | flag | off | |
| `--savedir` | str | `./` | |
| `--savefig` / `-S` | flag | off | |

Keep `fromfile_prefix_chars="@"` support so `@args.input` files still work,
consistent with other Yoke CLIs.

### 3.3 New entry point: `src/yoke/cli/plot_loss.py`

Thin `main()` matching `start_study.py` shape:
1. Build parser (`prog="yoke-plot-loss"`), call `cli.add_plot_loss_args(parser)`.
2. `args = parser.parse_args()`.
3. Call `plot_loss_curves(...)` then `save_or_show_figure(...)`.

Guarded by `if __name__ == "__main__": main()` (already excluded from coverage
via `pyproject.toml`).

### 3.4 Register the console script in `pyproject.toml`

Under `[project.scripts]`:

```toml
yoke-start-study = "yoke.cli.start_study:main"
yoke-plot-loss = "yoke.cli.plot_loss:main"
```

(Console-script name is `yoke-plot-loss`, see Resolved Decisions.)

## 4. Backwards compatibility for `applications/evaluation/TandVplot.py`

**Decision: keep a thin shim (Option A).** Replace the script body with a shim
that imports and calls `yoke.cli.plot_loss.main()`, preserving the old
invocation path (`python applications/evaluation/TandVplot.py @args.input`) so
existing user workflows/job scripts keep working. Removal can happen in a later
cleanup once users migrate to `yoke-plot-loss`.

## 5. Cleanups to fold in during the move

- Replace bare `print` warnings with `warnings.warn` (avoids noisy stdout).
  Note `-Werror` in CI: the non-divisible-remainder warning must be caught in
  tests via `pytest.warns(...)` so it does not fail the suite.
- Guard `trn_csv_list.pop()` when `--inprogress` and the list is empty.
- Remove commented-out dead code and the module-scope matplotlib backend lines.
- Choose a non-interactive backend (e.g. `Agg`) inside `save_or_show_figure`
  when `savefig` is set, so headless/CI runs never require a display.
- Add module + function docstrings and type annotations everywhere (`D`, `ANN`).

## 6. Tests (mirror `src` layout)

Add `tests/plots/__init__.py` and `tests/plots/test_loss_curves.py`:
- `find_record_csvs`: create temp `study_###` dirs with fake CSVs; assert file
  ordering, epoch parsing, and `inprogress` behavior (including empty-list edge).
- `compute_quantile_bands`: feed a known array; assert band shapes and the
  remainder-trimming path (assert the warning is emitted).
- `plot_loss_curves`: run with `matplotlib.use("Agg")`; assert it returns a
  `Figure` with expected axis labels / y-limit and >0 lines.
- `save_or_show_figure`: with `savefig=True` to `tmp_path`, assert PNG file is
  created at `study###_TandV_curve.png`.

Add `tests/cli/test_plot_loss.py` (mirrors `test_start_study.py`):
- `monkeypatch` `sys.argv`, point `--basedir` at a temp study dir, use
  `--savefig --savedir <tmp>`, call `plot_loss.main()`, assert the PNG exists.
- Confirm no display is required (Agg backend) so it passes in CI.

Ensure all new code is warning-clean under `pytest -Werror`.

## 7. Docs

- Update any harness/eval README references from
  `python applications/evaluation/TandVplot.py ...` to `yoke-plot-loss ...`.
- If Sphinx autodoc lists CLIs, add the new module under `docs/source/`.

## 8. Step-by-step implementation checklist

1. Create `src/yoke/plots/__init__.py` (module docstring).
2. Create `src/yoke/plots/loss_curves.py` with the functions in 3.1
   (ported/refactored from `TandVplot.py`, annotated + docstrings).
3. Add `add_plot_loss_args` to `src/yoke/helpers/cli.py` (3.2).
4. Create `src/yoke/cli/plot_loss.py` with `main()` (3.3).
5. Register `yoke-plot-loss` in `pyproject.toml` `[project.scripts]` (3.4).
6. Reduce `applications/evaluation/TandVplot.py` to a shim (or delete) (4).
7. Add tests under `tests/plots/` and `tests/cli/` (6).
8. Reinstall to register the new entry point:
   `flit install --symlink --deps develop`.
9. Run `pytest -Werror`, `ruff check`, `ruff format --check --diff`; fix.
10. Update docs/READMEs (7).

## 9. Validation commands

```bash
# Python env (confirmed): /Users/l255541/.conda/envs/yoke_260204/bin/python
flit install --symlink --deps develop            # register new console script
yoke-plot-loss --basedir ./runs --IDX 1 --savefig --savedir /tmp/tv
pytest -Werror tests/plots tests/cli
ruff check && ruff format --check --diff
```

## 10. Resolved decisions

1. **Console-script name:** `yoke-plot-loss`.
2. **Old script:** keep `applications/evaluation/TandVplot.py` as a thin shim
   for now (Option A) — it imports and calls `yoke.cli.plot_loss.main()`.
   Removal can happen in a later cleanup.
3. **Default `--basedir`:** change from `./study_directory` to `./runs` to match
   the harness `--rundir` default.
4. **Remainder message:** no preference given — implement with `warnings.warn`.
   Since CI runs `pytest -Werror`, tests must assert/catch the warning with
   `pytest.warns(...)` so it does not fail the suite.

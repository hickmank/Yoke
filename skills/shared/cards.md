# Handoff Cards

Each Yoke skill ends by writing a short **card** that the next skill reads.
Write the card in the skill's design notes file (`dev_plans/...`) under a
`## Card` heading, and show it in chat. Keep the field names exactly as
below, so a later skill (or a later session) can find them.

`yoke-new-harness` reads all three. `yoke-new-training-step` reads the
dataset card.

## Dataset card (from `yoke-new-dataset`)

```
Dataset card
- class:        yoke.datasets.<module>.<ClassName>
- module file:  src/yoke/datasets/<module>.py
- test file:    tests/datasets/test_<module>.py
- ctor args:    one per line, "name: type = default"; mark data paths [path]
- returns:      the __getitem__ tuple, each element with shape and dtype,
                e.g. (input_frames[2,C,H,W] float32, end_img[C,H,W] float32,
                      lead_times[2] float32)
- contract:     a row name from skills/shared/batch_contracts.md, or
                "NEW" (then the tuple above is the new contract)
- channels:     ordered channel names (these become the model's
                default_vars or channel_map)
- sample size:  e.g. image (H, W) = (1120, 400), half image
- __len__:      int(1e6) (random sampling) | N (fixed index)
- filelists:    applications/filelists/<name>_{train,validation,test}.txt
- data paths:   which CLI args carry them (e.g. --NPZ_DIR, --FILELIST_DIR)
- constraints:  e.g. "batch_size must be 1", "needs design CSV"
```

## Step card (from `yoke-new-training-step`)

```
Step card
- epoch fn:      yoke.utils.training.epoch.<module>.<function>
- datasteps:     train: yoke.utils.training.datastep.<module>.<function>
                 eval:  yoke.utils.training.datastep.<module>.<function>
- dataset tag:   DATASTEP_FN key passed as epoch_kwargs["dataset"], or "n/a"
- contract:      the batch tuple the datasteps unpack
- model call:    e.g. model(input_frames, in_vars, out_vars, lead_times)
- loss shape:    what loss_fn must return, e.g. (B, C) or elementwise
- loss builder:  suggested loss_builder for HarnessTrainer
- epoch_kwargs:  extra kwargs the harness must pass, with meaning
- records:       CSV columns written per batch, e.g. epoch, batch, loss
- tests:         test file paths
```

## Slurm card (from `yoke-slurm-template`)

```
Slurm card
- machine:       <ClusterName>
- profile:       skills/yoke-slurm-template/machines/<cluster>.md
- template:      path to training_slurm.tmpl (+ evaluation_slurm.tmpl if any)
- partition/qos: chosen values (and debug variant, if included)
- wall time:     per cycle
- CSV keys:      <KEY>s the template needs from the study CSV,
                 e.g. KNODES, NGPUS, allocation, train_script
- env:           environment activation line(s) used
- not verified:  anything that can only be checked by a real job
```

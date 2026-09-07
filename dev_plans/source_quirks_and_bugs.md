# Source Quirks & Bugs Found While Raising `src/yoke` Coverage

Status: **RESOLVED** — all three issues (plus sub-issue #2a) documented below have
been fixed and covered by tests. The full suite passes with `-Werror` and the
touched files pass `ruff check` / `ruff format`.

These were latent quirks/bugs in the installable package surfaced by coverage
work. They were originally logged (not fixed) to keep the coverage change set
focused; they have now been fixed in a dedicated change. Each section records the
original finding followed by a **Resolution** note.

---

## 1. `datasets/transforms.py` — `ResizePadCrop` crop logic is wrong (FIX SOON)

`ResizePadCrop.forward` crops using the wrong index of `pad_position` in two
places. `pad_position` is documented/asserted as `(dim0, dim1)` where:

- `pad_position[0]` ∈ {`"top"`, `"bottom"`} (vertical, dim -2)
- `pad_position[1]` ∈ {`"left"`, `"right"`} (horizontal, dim -1)

But the crop block (around `src/yoke/datasets/transforms.py:74`) reads:

```python
# Crop, ensuring we remove edges corresponding to the padding positions:
if self.pad_position[0] == "left":          # BUG: should be pad_position[1]
    img = img[..., -self.scaled_image_size[1] :]
else:
    img = img[..., : self.scaled_image_size[1]]
if self.pad_position[0] == "bottom":         # correct index, but see below
    img = img[..., -self.scaled_image_size[0] :, :]
else:
    img = img[..., : self.scaled_image_size[0], :]
```

Problems:

- **Line 75** checks `pad_position[0] == "left"`, but `pad_position[0]` is only
  ever `"top"`/`"bottom"` (enforced by the `assert` in `__init__`). So the
  horizontal (dim -1) crop branch that would keep the right edge is **unreachable
  dead code**, and the horizontal crop is effectively always "keep the leading
  slice", regardless of whether padding was requested on the left or right.
- The two `if`/`else` blocks are keyed on the wrong dimension for the horizontal
  crop; the horizontal crop should be driven by `pad_position[1]`.

**Impact:** when `pad_position[1] == "left"` (pad on the left), the transform still
crops from the left, so it removes real image content and keeps padding — the
opposite of the intent. Only the default `("bottom", "right")` path is exercised
in practice, which is why this hasn't bitten anyone yet.

**Suggested fix:** change the horizontal test to `self.pad_position[1] == "left"`
(and add a test asserting pixel-level correctness for all four
top/bottom × left/right combinations, not just output shape).

Coverage note: `transforms.py:76` is currently uncovered precisely because it is
unreachable given the `__init__` assertion.

**Resolution (FIXED):** the crop block now keys the horizontal crop on
`pad_position[1]` and the vertical crop on `pad_position[0]`, keeping the slice
away from the padded edge in every case (pad top -> keep bottom slice, pad left ->
keep right slice, etc.). Added `test_resize_pad_crop_pixel_correctness`, which
asserts pixel-level correctness (padding lands on the requested edges, original
content is preserved) across all four top/bottom × left/right combinations.

---

## 2. `utils/checkpointing.py` — HDF5 scalar params/buffers are saved but never
   reloaded (FIX SOON)

`save_model_and_optimizer_hdf5` stores 0-dim (scalar) parameters and buffers as
HDF5 **attributes** (see `src/yoke/utils/checkpointing.py:50` and `:57`):

```python
if data.ndim == 0:  # It's a scalar!
    h5f.attrs["model/parameters/" + name] = data
else:
    h5f.create_dataset("model/parameters/" + name, data=data)
```

But `load_model_and_optimizer_hdf5` only iterates the **group members**:

```python
for name in h5f.get("model/parameters", []):   # only dataset members, not attrs
    ...
for name in h5f.get("model/buffers", []):
    ...
```

**Impact:** any scalar (0-dim) parameter or buffer is silently **not restored** on
load — the freshly-constructed model keeps its initialized scalar value instead of
the checkpointed one. Confirmed by test: after a save/load round-trip, a scalar
`nn.Parameter` retained its constructor value rather than the trained value.

**Suggested fix:** on load, also iterate the relevant `h5f.attrs` keys (prefixed
`model/parameters/` and `model/buffers/`) and copy those scalars back into the
model, mirroring the save-side branch.

**Resolution (FIXED):** `load_model_and_optimizer_hdf5` now iterates
`h5f.attrs`, and for keys prefixed `model/parameters/` and `model/buffers/`
copies the scalar values back into the corresponding parameter/buffer. Verified
by round-trip value assertions in `tests/test_checkpointing.py`
(`test_hdf5_scalar_param_and_buffer_roundtrip`) and the strengthened
`tests/test_training_utils.py::test_hdf5_roundtrip_scalar_params_and_buffers`
(which now sets distinctive scalar values and asserts they survive the round
trip).

### 2a. HDF5 optimizer momentum state is likely not reloaded either

`save_model_and_optimizer_hdf5` writes optimizer state tensors under nested paths
like `optimizer/state{idx}/{k}` via `create_dataset`. The loader tries to find
them with:

```python
for name, group in h5f.items():        # top-level names only: "model", "optimizer"
    if "optimizer/state" in name:      # never matches a top-level name
        ...
```

`h5py.File.items()` yields only **top-level** group names (`"model"`,
`"optimizer"`), so `"optimizer/state" in name` never matches and the momentum-load
loop (`src/yoke/utils/checkpointing.py:145`–`150`) does not execute. Per-parameter
optimizer state (e.g. SGD momentum buffers) therefore appears **not** to be
restored from HDF5 checkpoints.

**Suggested fix:** descend into the `optimizer` group (e.g. iterate
`h5f["optimizer"].items()` or walk with `h5f.visititems`) and match the
`state{idx}` subgroups there. Add a test that asserts optimizer momentum buffers
survive an HDF5 round-trip (not just shape/keys).

Coverage note: `checkpointing.py` lines `100, 111, 123, 145-150` are uncovered
because of the two issues above (dead/unreachable load branches); lines
`195, 202, 225, 255, 268-270, 303` are the `dist.is_initialized()` DDP branches,
which need a live process group to exercise.

**Resolution (FIXED):** the loader now descends into the `optimizer` group and
matches `state{idx}` subgroups there. Because a freshly-constructed optimizer has
an empty `state` dict, the entries are now built directly keyed by the saved
parameter index (rather than indexing into an assumed-populated `state` dict,
which raised `IndexError`). Scalar (0-dim) state datasets (e.g. the SGD
`momentum_buffer` of a scalar parameter) are read with `[()]` and wrapped in
`np.asarray` so `torch.from_numpy` accepts them. Verified by
`test_hdf5_optimizer_momentum_roundtrip`, which asserts SGD momentum buffers
survive the round trip.

---

## 3. `helpers/strings.py` — `bool` branch is unreachable for Python `bool`

`replace_keys` orders its type checks as int-before-bool:

```python
elif isinstance(value, np.int64) or isinstance(value, int):   # matches bool too!
    data = data.replace(f"<{key}>", f"{value:d}")
...
elif isinstance(value, np.bool_) or isinstance(value, bool):
    data = data.replace(f"<{key}>", f"{str(value)}")
```

Since Python `bool` is a subclass of `int`, a plain `True`/`False` is formatted by
the **int** branch (`f"{value:d}"` -> `"1"`/`"0"`), never reaching the bool branch.
Only a NumPy `np.bool_` reaches the dedicated bool branch (rendering `"True"`/
`"False"`).

**Impact:** low/cosmetic — a CSV value of Python `True` renders as `"1"` rather
than `"True"`. This is more of an inconsistency than a bug, but worth noting: the
same logical value renders differently depending on whether it arrives as a Python
`bool` or a `np.bool_`. If a canonical rendering is desired, move the bool check
above the int check.

This one is **not urgent**; documented for awareness. The new `strings` tests
cover both the Python-`bool` (int path) and `np.bool_` (bool path) behaviors as
they currently stand.

**Resolution (FIXED):** the bool check now precedes the int check in
`replace_keys`, so both Python `bool` and `np.bool_` render canonically as
`"True"`/`"False"`. The `test_replace_keys` parametrization was updated
accordingly (Python `True` -> `"True"`, and a new `False` -> `"False"` case).

---

## Recommended follow-up

- [x] Fix `ResizePadCrop` horizontal crop indexing (#1) + add pixel-level tests.
- [x] Fix HDF5 scalar param/buffer reload (#2) and optimizer-state reload (#2a) +
      add round-trip value assertions.
- [x] Decide on canonical bool rendering in `replace_keys` (#3) — rendered as
      `"True"`/`"False"` for both Python `bool` and `np.bool_`.

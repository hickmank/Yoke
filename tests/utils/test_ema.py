"""Tests for the warmup-EMA utilities in :mod:`yoke.utils.ema`.

Covers the Diffusers warmup decay schedule, the ``AveragedModel`` copy-then-
average timing, end-to-end training that produces distinct EMA weights, the
save/load round-trip into a fresh model, and a continuation (save/restore of the
EMA state plus global step counter) cycle.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from yoke.utils.ema import (
    build_ema_model,
    compute_warmup_decay,
    load_ema_into_model,
    make_ema_hooks,
    make_warmup_ema_fn,
    save_ema_checkpoint,
)


class TinyNet(nn.Module):
    """Minimal model for EMA testing (LayerNorm, not BatchNorm)."""

    def __init__(self, dim: int = 8) -> None:
        """Initialization."""
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map."""
        return self.norm(self.linear(x))


# ============================================================================
# Schedule tests (Section 2.6)
# ============================================================================


@pytest.mark.parametrize("s", [1, 2, 5, 10, 100])
def test_warmup_decay_matches_formula(s: int) -> None:
    """Decay matches min(0.9999, 1 - (1 + s)^(-2/3)) for the reference config."""
    beta = compute_warmup_decay(
        s, max_decay=0.9999, inv_gamma=1.0, power=2.0 / 3.0, min_decay=0.0
    )
    expected = min(0.9999, 1.0 - (1.0 + s) ** (-2.0 / 3.0))
    assert beta == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_warmup_decay_zero_step_is_zero() -> None:
    """At s=0 the decay is 0 (pure copy of model params)."""
    assert compute_warmup_decay(0) == pytest.approx(0.0)


def test_warmup_decay_clamped_to_max() -> None:
    """Decay is clamped to max_decay for very large step counts."""
    beta = compute_warmup_decay(10**9, max_decay=0.9999)
    assert beta == pytest.approx(0.9999)


def test_warmup_ema_fn_in_place_update() -> None:
    """The multi_avg_fn updates EMA params in place: ema*beta + model*(1-beta)."""
    fn = make_warmup_ema_fn(max_decay=0.9999, inv_gamma=1.0, power=2.0 / 3.0)
    ema_p = [torch.ones(3)]
    model_p = [torch.zeros(3)]
    num_averaged = 1
    beta = compute_warmup_decay(num_averaged)

    fn(ema_p, model_p, num_averaged)

    # ema = 1*beta + 0*(1-beta) = beta
    assert torch.allclose(ema_p[0], torch.full((3,), beta), atol=1e-7)


# ============================================================================
# Timing tests (Section 2.3)
# ============================================================================


def test_first_update_copies_model() -> None:
    """The first update_parameters call copies the model (num_averaged 0 -> 1)."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)

    assert int(ema.n_averaged.item()) == 0

    # First call copies the model weights exactly.
    ema.update_parameters(model)
    assert int(ema.n_averaged.item()) == 1

    for p_ema, p_model in zip(ema.module.parameters(), model.parameters()):
        assert torch.allclose(p_ema, p_model)


def test_second_update_uses_step_one() -> None:
    """The second update averages with s=1 (the first nonzero Diffusers decay)."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)

    # First call: copy.
    ema.update_parameters(model)
    ema_after_copy = [p.detach().clone() for p in ema.module.parameters()]

    # Mutate the model so the next average is observable.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Second call: s = num_averaged = 1 at the time the avg_fn runs.
    beta = compute_warmup_decay(1)
    ema.update_parameters(model)

    for p_ema, p_copy, p_model in zip(
        ema.module.parameters(), ema_after_copy, model.parameters()
    ):
        expected = p_copy * beta + p_model * (1.0 - beta)
        assert torch.allclose(p_ema, expected, atol=1e-6)


# ============================================================================
# End-to-end tests (Section 2.6)
# ============================================================================


def test_ema_differs_from_raw_after_training() -> None:
    """After several steps EMA params differ from the raw model params."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(10):
        x = torch.randn(16, 8)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(model)

    differs = any(
        not torch.allclose(p_ema, p_model)
        for p_ema, p_model in zip(ema.module.parameters(), model.parameters())
    )
    assert differs


def test_ema_save_load_roundtrip() -> None:
    """save_ema_checkpoint -> load_ema_into_model reproduces EMA weights."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(5):
        loss = model(torch.randn(16, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(model)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ema.pth")
        save_ema_checkpoint(ema, path)

        fresh = TinyNet()
        load_ema_into_model(fresh, path)

    for p_fresh, p_ema in zip(fresh.parameters(), ema.module.parameters()):
        assert torch.allclose(p_fresh, p_ema)


def test_ema_continuation_state_survives_restore() -> None:
    """EMA state_dict + global step counter survive a save/restore cycle."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = build_ema_model(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    global_step = 0
    for _ in range(7):
        loss = model(torch.randn(16, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        global_step += 1
        ema.update_parameters(model)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ema_state.pth")
        torch.save(
            {"ema_state_dict": ema.state_dict(), "global_step": global_step},
            path,
        )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model2 = TinyNet()
        ema2 = build_ema_model(model2)
        ema2.load_state_dict(ckpt["ema_state_dict"])

    assert ckpt["global_step"] == global_step
    assert int(ema2.n_averaged.item()) == int(ema.n_averaged.item())
    for p2, p in zip(ema2.module.parameters(), ema.module.parameters()):
        assert torch.allclose(p2, p)


# ============================================================================
# make_ema_hooks tests (HarnessTrainer integration)
# ============================================================================


class _DDPLike(nn.Module):
    """Stand-in for DDP that exposes ``.module`` like the real wrapper."""

    def __init__(self, module: nn.Module) -> None:
        """Wrap ``module`` and expose it as ``.module``."""
        super().__init__()
        self.module = module

    def forward(self, *a: object, **k: object) -> object:
        """Delegate to the wrapped module."""
        return self.module(*a, **k)


def _fake_trainer(**overrides: object) -> SimpleNamespace:
    """Build a minimal trainer-like namespace for hook testing."""
    model = TinyNet()
    base = dict(
        model=_DDPLike(model),
        device=torch.device("cpu"),
        args=SimpleNamespace(continuation=False, checkpoint=None, anchor_lr=1e-4),
        starting_epoch=0,
        global_step=0,
        rank=0,
        model_args={"dim": 8},
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        new_chkpt_path=None,
        ema_model=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_make_ema_hooks_fresh_builds_shadow() -> None:
    """on_after_ddp_wrap builds an EMA shadow and registers it on the trainer."""
    on_after_ddp_wrap, _ = make_ema_hooks(TinyNet)
    trainer = _fake_trainer()

    on_after_ddp_wrap(trainer)

    assert trainer.ema_model is not None
    # Fresh EMA shadow mirrors the underlying module weights before any update.
    for p_ema, p_model in zip(
        trainer.ema_model.module.parameters(), trainer.model.module.parameters()
    ):
        assert torch.allclose(p_ema, p_model)
    # Fresh run leaves global_step untouched.
    assert trainer.global_step == 0


def test_make_ema_hooks_save_writes_companion_and_returns_extra_state() -> None:
    """on_before_save writes companion + production files and returns global_step."""
    on_after_ddp_wrap, on_before_save = make_ema_hooks(TinyNet)

    with tempfile.TemporaryDirectory() as tmp:
        main_path = os.path.join(tmp, "study001_modelState_epoch0003.pth")
        trainer = _fake_trainer(new_chkpt_path=main_path, global_step=1234)
        on_after_ddp_wrap(trainer)

        extra = on_before_save(trainer, 3)

        assert extra == {"global_step": 1234}
        assert os.path.exists(main_path.replace(".pth", "_ema.pth"))
        assert os.path.exists(main_path.replace(".pth", "_ema_weights.pth"))


def test_make_ema_hooks_save_no_shadow_returns_none() -> None:
    """on_before_save is a no-op returning None when no EMA shadow exists."""
    _, on_before_save = make_ema_hooks(TinyNet)
    trainer = _fake_trainer(ema_model=None, new_chkpt_path="unused.pth")

    assert on_before_save(trainer, 1) is None


def test_make_ema_hooks_continuation_restores_state() -> None:
    """On continuation the shadow + global_step are restored from the companion."""
    on_after_ddp_wrap, on_before_save = make_ema_hooks(TinyNet)

    with tempfile.TemporaryDirectory() as tmp:
        main_path = os.path.join(tmp, "study001_modelState_epoch0002.pth")

        # First, produce a companion checkpoint at epoch 2 with a known step.
        src = _fake_trainer(new_chkpt_path=main_path, global_step=555)
        on_after_ddp_wrap(src)
        # Perturb the shadow so restoration is observable.
        with torch.no_grad():
            for p in src.ema_model.module.parameters():
                p.add_(0.5)
        on_before_save(src, 2)

        # Now continue: point args.checkpoint at the main path (epoch 2).
        dst = _fake_trainer(
            args=SimpleNamespace(
                continuation=True, checkpoint=main_path, anchor_lr=1e-4
            ),
            starting_epoch=2,
        )
        on_after_ddp_wrap(dst)

        assert dst.global_step == 555
        for p_dst, p_src in zip(
            dst.ema_model.module.parameters(), src.ema_model.module.parameters()
        ):
            assert torch.allclose(p_dst, p_src)


def test_make_ema_hooks_continuation_epoch_mismatch_raises() -> None:
    """A companion epoch that disagrees with starting_epoch raises ValueError."""
    on_after_ddp_wrap, on_before_save = make_ema_hooks(TinyNet)

    with tempfile.TemporaryDirectory() as tmp:
        main_path = os.path.join(tmp, "study001_modelState_epoch0002.pth")
        src = _fake_trainer(new_chkpt_path=main_path, global_step=10)
        on_after_ddp_wrap(src)
        on_before_save(src, 2)  # companion epoch = 2

        dst = _fake_trainer(
            args=SimpleNamespace(
                continuation=True, checkpoint=main_path, anchor_lr=1e-4
            ),
            starting_epoch=5,  # disagrees with companion epoch 2
        )
        with pytest.raises(ValueError, match="does not match"):
            on_after_ddp_wrap(dst)


def test_make_ema_hooks_continuation_missing_companion_starts_fresh() -> None:
    """A missing companion checkpoint starts EMA fresh without error."""
    on_after_ddp_wrap, _ = make_ema_hooks(TinyNet)
    dst = _fake_trainer(
        args=SimpleNamespace(
            continuation=True, checkpoint="/nonexistent/ckpt.pth", anchor_lr=1e-4
        ),
        starting_epoch=3,
    )

    on_after_ddp_wrap(dst)

    assert dst.ema_model is not None
    assert dst.global_step == 0


def test_make_ema_hooks_continuation_zero_step_warns(
    capsys: pytest.CaptureFixture,
) -> None:
    """Resuming at epoch>0 with a persisted global_step of 0 warns (no raise)."""
    on_after_ddp_wrap, on_before_save = make_ema_hooks(TinyNet)

    with tempfile.TemporaryDirectory() as tmp:
        main_path = os.path.join(tmp, "study001_modelState_epoch0002.pth")
        # Companion written at epoch 2 with global_step 0.
        src = _fake_trainer(new_chkpt_path=main_path, global_step=0)
        on_after_ddp_wrap(src)
        on_before_save(src, 2)

        dst = _fake_trainer(
            args=SimpleNamespace(
                continuation=True, checkpoint=main_path, anchor_lr=1e-4
            ),
            starting_epoch=2,
        )
        on_after_ddp_wrap(dst)

    assert dst.global_step == 0
    assert "global_step is 0 while resuming" in capsys.readouterr().out

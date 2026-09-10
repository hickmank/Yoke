Authoring a harness with ``HarnessTrainer``
===========================================

Historically every training harness under ``applications/harnesses/`` shipped a
``train_*.py`` script that was, in the large, the same program copied and
lightly edited. :class:`yoke.harnesses.trainer.HarnessTrainer` removes that
copy-paste by owning the fixed *orchestration* (DDP setup, model/optimizer/
scheduler/dataloader construction, a timed epoch loop, ``.pth`` checkpointing,
and job resubmission) while a thin harness script injects only the parts that
genuinely vary.

The trainer is **always DDP** and **always writes** ``.pth`` checkpoints via
:func:`yoke.utils.checkpointing.save_model_and_optimizer`. Vanilla
``DataParallel`` and HDF5 checkpoint *writing* are deprecated (targeted for
removal in December 2026).

What you inject
---------------

- ``model_builder(args, device) -> (model, model_args, model_class, start_epoch, optimizer)``
  — constructs the model for **both** the fresh and continuation paths. On a
  fresh run it returns the constructed model (already on ``device``), the
  ``model_args`` dict, the model *class* (so the checkpoint records a matching
  ``model_class``), ``0`` for the starting epoch, and ``None`` for the optimizer
  (the trainer then builds one). On continuation it returns a *restored*
  optimizer so its state is preserved. For the common fresh-vs-continue case use
  the :func:`yoke.utils.builders.build_from_checkpoint` factory; write a bespoke
  builder when performing architectural surgery (loading a pretrained backbone,
  stripping/replacing layers, etc.).
- ``dataset_builder(args) -> (train_dataset, val_dataset)``.
- ``epoch_fn`` — one of the existing per-epoch functions in
  :mod:`yoke.utils.training.epoch`. Study-specific extra keyword arguments (e.g.
  ``channel_map`` or a ``dataset`` tag) are supplied via ``epoch_kwargs``.
- ``optimizer_builder(model, args) -> Optimizer`` — defaults to
  :func:`yoke.utils.builders.build_adamw`.
- ``scheduler_builder(optimizer, args, last_epoch) -> scheduler | None`` —
  optional.
- ``loss_builder() -> nn.Module`` — defaults to
  :func:`yoke.utils.builders.default_mse_loss`.

Dynamic, in-loop customization
-------------------------------

EMA, gradient clipping, and progressive (un)freezing are expressed through a
small :class:`yoke.harnesses.trainer.TrainerHooks` object of optional,
composable callables: ``on_after_ddp_wrap``, ``on_epoch_start``,
``on_before_optimizer_step``, ``on_after_step``, and ``on_before_save`` (which
may return a ``dict`` merged into the checkpoint's ``extra_state``). Because
hooks compose, a study needing EMA *and* grad-clip *and* scheduled unfreezing
combines them without any multiple-inheritance mess.

A thin harness script
----------------------

.. code-block:: python

    if __name__ == "__main__":
        args = parser.parse_args()

        HarnessTrainer(
            args,
            model_builder=build_from_checkpoint(LodeRunner, make_model_args),
            dataset_builder=build_dataset,
            epoch_fn=train_DDP_loderunner_epoch,
            scheduler_builder=build_scheduler,
            epoch_kwargs={"channel_map": list(range(len(CHANNEL_LIST)))},
        ).run()

See ``applications/harnesses/se_DDP_loderunner/train_LodeRunner_ddp.py`` for a
complete migrated example.

Demo harnesses
--------------

``moving_mnist`` and ``mnist_surrogate`` are intentionally kept as small,
single-process, bespoke reference scripts and do **not** use ``HarnessTrainer``.
They are minimal onboarding examples; migrating them would reintroduce a
single-process path that the "always DDP" design deliberately removes.

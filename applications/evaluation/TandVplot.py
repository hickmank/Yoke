"""Backwards-compatible shim for the Yoke loss-curve plotting CLI.

This script previously contained the full implementation for plotting Yoke
training/validation learning curves. That logic now lives in
``yoke.plots.loss_curves`` and is exposed as the installed console script
``yoke-plot-loss`` (see ``yoke.cli.plot_loss``).

This shim is retained so existing workflows that invoke the script directly
(e.g. ``python applications/evaluation/TandVplot.py @args.input``) keep working.
Prefer ``yoke-plot-loss`` for new usage. This shim may be removed in a future
cleanup.
"""

from yoke.cli.plot_loss import main


if __name__ == "__main__":
    main()

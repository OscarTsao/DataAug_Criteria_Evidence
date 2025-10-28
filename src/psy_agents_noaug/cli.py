"""CLI entrypoints for the PSY Agents NO‑AUG package.

The CLI offers a thin, dependency-light interface for:
  - training (stubbed here; wire to your project trainer)
  - hyperparameter optimisation (delegates to scripts/tune_max.py)
  - printing top‑K HPO results exported by the tuner script

Design goals:
  - Keep the surface simple and self-documenting
  - Avoid importing heavy frameworks until subcommands run
  - Be explicit about side effects (e.g., MLflow env vars)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer

# Top-level app: subcommands are declared below.
app = typer.Typer(
    help="NoAug Criteria/Evidence: train, tune (HPO), eval, and show-best."
)


def _default_outdir(outdir: str | None) -> str:
    """Return the provided output directory or a sensible default.

    We keep default runtime artifacts under ./_runs to avoid polluting the
    repository tree and to make it easy to clean up experiments locally.
    """
    return outdir or "./_runs"


def _ensure_mlflow(outdir: str) -> None:
    """Ensure MLflow writes to an isolated file store under ``outdir``.

    The CLI uses a file URI to keep experiments local by default. Users can
    still override via environment variables when needed.
    """
    os.makedirs(outdir, exist_ok=True)
    mlruns = Path(outdir) / "mlruns"
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{mlruns.as_posix()}")


@app.command()
def train(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    model_name: str = typer.Option("bert-base-uncased"),
    outdir: str | None = typer.Option(None),
    epochs: int = typer.Option(3),
    seed: int = typer.Option(42),
    batch_size: int = typer.Option(16),
    grad_accum: int = typer.Option(1),
    config: str | None = typer.Option(
        None, help="Optional JSON config to override defaults"
    ),
    aug_lib: str = typer.Option(
        "none",
        "--aug-lib",
        help="Augmentation library: none|nlpaug|textattack|both",
    ),
    aug_methods: str = typer.Option(
        "all",
        "--aug-methods",
        help="Comma separated augmenter names or 'all'",
    ),
    aug_p_apply: float = typer.Option(0.15, "--aug-p-apply"),
    aug_ops_per_sample: int = typer.Option(1, "--aug-ops-per-sample"),
    aug_max_replace: float = typer.Option(0.3, "--aug-max-replace"),
    aug_tfidf_model: str | None = typer.Option(None, "--aug-tfidf-model"),
    aug_reserved_map: str | None = typer.Option(None, "--aug-reserved-map"),
    loader_workers: int | None = typer.Option(None, "--loader-workers"),
    prefetch_factor: int | None = typer.Option(None, "--prefetch-factor"),
):
    """Run a training job.

    Notes
    -----
    This command is intentionally thin to keep the CLI fast to import and
    test. It prints the parsed configuration and environment so you can wire
    it to your training entrypoint as needed for your environment.
    """
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)

    methods = [m.strip() for m in aug_methods.split(",") if m.strip()]
    if not methods:
        methods = ["all"]

    typer.echo(
        f"[train] agent={agent} model={model_name} epochs={epochs} seed={seed} outdir={outdir}"
    )
    typer.echo(
        f"[train] augment lib={aug_lib} methods={methods} p_apply={aug_p_apply:.2f} ops={aug_ops_per_sample} max_replace={aug_max_replace:.2f}"
    )
    if loader_workers is not None or prefetch_factor is not None:
        typer.echo(
            f"[train] dataloader workers={loader_workers} prefetch={prefetch_factor}"
        )
    if aug_tfidf_model:
        typer.echo(f"[train] aug tfidf model={aug_tfidf_model}")
    if aug_reserved_map:
        typer.echo(f"[train] aug reserved map={aug_reserved_map}")
    if config:
        cfg = json.loads(Path(config).read_text())
        typer.echo(f"[train] loaded config keys: {list(cfg.keys())}")


@app.command()
def tune(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    study: str = typer.Option(...),
    n_trials: int = typer.Option(200),
    timeout: int | None = typer.Option(None),
    parallel: int = typer.Option(1),
    outdir: str | None = typer.Option(None),
    storage: str | None = typer.Option(
        None, help="Optuna storage URL (e.g., sqlite:///path/to.db)"
    ),
    multi_objective: bool = typer.Option(False),
    from_best_of: str | None = typer.Option(
        None, "--from-best-of", help="Seed augmentation sweep from another study"
    ),
    hpo_augment_only: bool = typer.Option(
        False, "--hpo-augment-only", help="Search augmentation params only"
    ),
    aug_lib: str = typer.Option("none", "--aug-lib"),
    aug_methods: str = typer.Option("all", "--aug-methods"),
    aug_p_apply: float = typer.Option(0.15, "--aug-p-apply"),
    aug_ops_per_sample: int = typer.Option(1, "--aug-ops-per-sample"),
    aug_max_replace: float = typer.Option(0.3, "--aug-max-replace"),
):
    """Launch maximal HPO via ``scripts/tune_max.py``.

    The CLI merely marshals arguments and environment. Search spaces and the
    training bridge live inside the script to keep imports isolated.
    """
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)
    storage = storage or f"sqlite:///{Path('./_optuna/noaug.db').absolute()}"
    typer.echo(
        f"[tune] agent={agent} study={study} augment-lib={aug_lib} methods={aug_methods}"
    )
    if from_best_of:
        typer.echo(f"[tune] initializing from study={from_best_of}")
    if hpo_augment_only:
        typer.echo("[tune] restricting search space to augmentation parameters")

    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent",
        agent,
        "--study",
        study,
        "--n-trials",
        str(n_trials),
        "--parallel",
        str(parallel),
        "--outdir",
        outdir,
        "--storage",
        storage,
    ]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    if multi_objective:
        cmd += ["--multi-objective"]
    subprocess.run(cmd, check=True)


@app.command("show-best")
def show_best(
    agent: str = typer.Option(...),
    study: str = typer.Option(...),
    outdir: str | None = typer.Option(None),
    topk: int = typer.Option(5),
):
    """Pretty‑print top‑K trials exported by ``tune_max.py``.

    Expects a JSON file produced by the tuner at ``{outdir}/{agent}_{study}_topk.json``.
    """
    outdir = _default_outdir(outdir)
    path = Path(outdir) / f"{agent}_{study}_topk.json"
    if not path.exists():
        typer.echo(f"Not found: {path}")
        raise typer.Exit(1)
    data = json.loads(path.read_text())
    for i, t in enumerate(data[:topk], 1):
        val = t.get("value")
        params = t.get("params", {})
        typer.echo(f"[{i}] value={val:.4f}  params={json.dumps(params)[:500]}...")


@app.command("tune-supermax")
def tune_supermax(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    study: str = typer.Option(...),
    n_trials: int = typer.Option(5000, help="Very large default; override as needed"),
    parallel: int = typer.Option(4),
    outdir: str | None = typer.Option(None),
    storage: str | None = typer.Option(None),
):
    """Run very large HPO trials suitable for long-running servers.

    Configures 100-epoch trials and patience-based early stopping via env vars
    so downstream code does not need to be modified.
    """
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)
    storage = storage or f"sqlite:///{Path('./_optuna/noaug.db').absolute()}"
    env = os.environ.copy()
    env["HPO_EPOCHS"] = "100"
    env["HPO_PATIENCE"] = "20"
    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent",
        agent,
        "--study",
        study,
        "--n-trials",
        str(n_trials),
        "--parallel",
        str(parallel),
        "--outdir",
        outdir,
        "--storage",
        storage,
    ]
    subprocess.run(cmd, check=True, env=env)


def main():
    app()


if __name__ == "__main__":
    main()

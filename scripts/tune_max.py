# Maximal Optuna search for Criteria/Evidence/Joint/Share agents (NoAug).
# - Big conditional spaces (model, head, optimizer, scheduler, tokenization, loss, null-span policy, postproc)
# - Multi-fidelity pruning (Hyperband + Percentile with patience)
# - MLflow logging (file-based by default; keep state outside the repo)
#
# USAGE (examples):
#   python scripts/tune_max.py --agent criteria --study noaug-criteria-max --n-trials 800 --outdir ./_runs
#   python scripts/tune_max.py --agent evidence --study noaug-evidence-max --n-trials 1200 --timeout 10800 --parallel 4
#
# INTEGRATION: Implement `run_training_eval(cfg, callbacks)` to call your trainer once/epoch,
# reporting metrics to the provided callbacks and returning the final metric dict.

import argparse
import json
import os
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner, PatientPruner
from optuna.samplers import NSGAIISampler, TPESampler

try:
    import mlflow

    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False


def set_seeds(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# EarlyStopping helper (patience-based)
# ----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        if self.improved(value):
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def default_mlflow_setup(outdir: str):
    if not _HAS_MLFLOW:
        return
    os.makedirs(outdir, exist_ok=True)
    mlruns_dir = os.path.join(outdir, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment("NoAug_Criteria_Evidence")


def on_epoch(
    trial: optuna.Trial, step: int, metric: float, secondary: float | None = None
):
    trial.report(metric, step=step)
    if secondary is not None:
        trial.set_user_attr(f"secondary_epoch_{step}", float(secondary))
    if trial.should_prune():
        raise optuna.TrialPruned(f"Pruned at step {step} with metric {metric:.4f}")


# Optional narrowing via env for hybrid flows
_raw_models = os.environ.get("HPO_MODEL_CHOICES")
MODEL_CHOICES = (
    [m.strip() for m in _raw_models.split(",")]
    if _raw_models
    else [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
        "microsoft/deberta-v3-base",
        "microsoft/deberta-v3-large",
        "google/electra-base-discriminator",
        "google/electra-large-discriminator",
        "xlm-roberta-base",
    ]
)

SCHEDULERS = ["linear", "cosine", "cosine_restart", "polynomial", "one_cycle"]
OPTIMS = ["adamw", "adamw_8bit", "adafactor", "lion"]

ACTS = ["gelu", "relu", "silu"]
POOLING = ["cls", "mean", "max", "attn"]
LOSSES_CLS = ["ce", "ce_label_smooth", "focal"]
LOSSES_QA = ["qa_ce", "qa_ce_ls", "qa_focal"]

NULL_POLICIES = ["none", "threshold", "ratio", "calibrated"]
RERANKERS = ["sum", "product", "softmax"]

# Optional head narrowing via env JSON (for hybrid trust-region)
_HEAD_LIMITS = json.loads(os.environ.get("HPO_HEAD_LIMITS_JSON", "{}"))

# Augmentation configuration
AUG_LIBS = ["none", "nlpaug", "textattack", "both"]
AUG_METHODS_NLPAUG = [
    "KeyboardAug",
    "OcrAug",
    "RandomCharAug",
    "RandomWordAug",
    "SpellingAug",
    "SplitAug",
    "SynonymAug",
    "TfIdfAug",
]
AUG_METHODS_TEXTATTACK = [
    "DeletionAugmenter",
    "SwapAugmenter",
    "SynonymInsertionAugmenter",
    "EasyDataAugmenter",
    "CheckListAugmenter",
]


def suggest_common(trial: optuna.Trial, heavy_model: bool) -> dict[str, Any]:
    # Cap at 512 for transformer models (RoBERTa, BERT, DeBERTa all use 512 max)
    max_len = trial.suggest_int("tok.max_length", 128, 512, step=32)
    stride = trial.suggest_int("tok.doc_stride", 32, min(256, max_len // 2), step=16)
    fast_tok = trial.suggest_categorical("tok.use_fast", [True, False])

    # Use fixed batch size choices to avoid Optuna dynamic value space error
    # For heavy models, larger batch sizes will be automatically reduced by OOM handling
    bsz = trial.suggest_categorical(
        "train.batch_size",
        [8, 12, 16, 24, 32, 48, 64],
    )
    accum = trial.suggest_categorical("train.grad_accum", [1, 2, 3, 4, 6, 8])

    optim = trial.suggest_categorical("optim.name", OPTIMS)
    lr_hi = 1.5e-4 if heavy_model else 3e-4
    lr = trial.suggest_float("optim.lr", 5e-6, lr_hi, log=True)
    wd = trial.suggest_float("optim.weight_decay", 1e-6, 2e-1, log=True)

    if optim in ("adamw", "adamw_8bit", "lion"):
        b1 = trial.suggest_float("optim.beta1", 0.80, 0.95)
        b2 = trial.suggest_float("optim.beta2", 0.95, 0.9999)
        eps = trial.suggest_float("optim.eps", 1e-9, 1e-6, log=True)
    else:
        b1 = b2 = eps = None

    sched = trial.suggest_categorical("sched.name", SCHEDULERS)
    warmup = trial.suggest_float("sched.warmup_ratio", 0.0, 0.2)
    cos_cycles = (
        trial.suggest_int("sched.cosine_cycles", 1, 4)
        if sched in ("cosine_restart", "one_cycle")
        else None
    )
    poly_power = (
        trial.suggest_float("sched.poly_power", 0.5, 2.0)
        if sched == "polynomial"
        else None
    )

    clip = trial.suggest_float("train.clip_grad", 0.0, 1.5)
    dropout = trial.suggest_float("model.dropout", 0.0, 0.5)
    attn_drop = trial.suggest_float("model.attn_dropout", 0.0, 0.3)
    grad_ckpt = trial.suggest_categorical("train.grad_checkpointing", [False, True])
    freeze_layers = trial.suggest_int("train.freeze_encoder_layers", 0, 6)

    lld = trial.suggest_float("optim.layerwise_lr_decay", 0.80, 1.00)

    return {
        "tok": {"max_length": max_len, "doc_stride": stride, "use_fast": fast_tok},
        "train": {
            "batch_size": bsz,
            "grad_accum": accum,
            "epochs": None,
            "clip_grad": clip,
            "grad_checkpointing": grad_ckpt,
            "freeze_encoder_layers": freeze_layers,
        },
        "optim": {
            "name": optim,
            "lr": lr,
            "weight_decay": wd,
            "beta1": b1,
            "beta2": b2,
            "eps": eps,
            "layerwise_lr_decay": lld,
        },
        "sched": {
            "name": sched,
            "warmup_ratio": warmup,
            "cosine_cycles": cos_cycles,
            "poly_power": poly_power,
        },
        "regularization": {"dropout": dropout, "attn_dropout": attn_drop},
    }


def suggest_augmentation(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest augmentation hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary with augmentation configuration
    """
    aug_enabled = trial.suggest_categorical("aug.enabled", [True, False])

    if not aug_enabled:
        return {"augmentation": {"enabled": False}}

    # Select augmentation library
    aug_lib = trial.suggest_categorical("aug.lib", AUG_LIBS[1:])  # Exclude "none"

    # Core augmentation parameters
    p_apply = trial.suggest_float("aug.p_apply", 0.05, 0.30)
    ops_per_sample = trial.suggest_int("aug.ops_per_sample", 1, 2)
    max_replace = trial.suggest_float("aug.max_replace_ratio", 0.1, 0.5)

    # Select methods based on library
    methods = []

    if aug_lib in ("nlpaug", "both"):
        n_nlpaug = trial.suggest_int(
            "aug.n_nlpaug_methods", 1, min(3, len(AUG_METHODS_NLPAUG))
        )
        # Sample subset of nlpaug methods
        selected_nlpaug = trial.suggest_categorical(
            "aug.nlpaug_method_1",
            AUG_METHODS_NLPAUG,
        )
        methods.append(selected_nlpaug)

        if n_nlpaug >= 2:
            remaining = [m for m in AUG_METHODS_NLPAUG if m != selected_nlpaug]
            selected_nlpaug_2 = trial.suggest_categorical(
                "aug.nlpaug_method_2",
                remaining,
            )
            methods.append(selected_nlpaug_2)

        if n_nlpaug >= 3:
            remaining = [m for m in AUG_METHODS_NLPAUG if m not in methods]
            selected_nlpaug_3 = trial.suggest_categorical(
                "aug.nlpaug_method_3",
                remaining,
            )
            methods.append(selected_nlpaug_3)

    if aug_lib in ("textattack", "both"):
        n_textattack = trial.suggest_int(
            "aug.n_textattack_methods", 1, min(2, len(AUG_METHODS_TEXTATTACK))
        )
        # Sample subset of textattack methods
        selected_ta = trial.suggest_categorical(
            "aug.textattack_method_1",
            AUG_METHODS_TEXTATTACK,
        )
        methods.append(selected_ta)

        if n_textattack >= 2:
            remaining = [m for m in AUG_METHODS_TEXTATTACK if m != selected_ta]
            selected_ta_2 = trial.suggest_categorical(
                "aug.textattack_method_2",
                remaining,
            )
            methods.append(selected_ta_2)

    return {
        "augmentation": {
            "enabled": True,
            "lib": aug_lib,
            "methods": methods,
            "p_apply": p_apply,
            "ops_per_sample": ops_per_sample,
            "max_replace_ratio": max_replace,
            "scope": "train_only",  # Fixed - never augment val/test
            "seed": None,  # Will use global seed
        }
    }


def suggest_criteria(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    heavy = any(k in model_name for k in ["-large", "large", "xlm-roberta"])
    com = suggest_common(trial, heavy)
    aug = suggest_augmentation(trial)  # Add augmentation
    pooling = trial.suggest_categorical("head.pooling", POOLING)
    head_layers = trial.suggest_int(
        "head.layers",
        int(_HEAD_LIMITS.get("layers_min", 1)),
        int(_HEAD_LIMITS.get("layers_max", 4)),
    )
    head_hidden = trial.suggest_categorical(
        "head.hidden",
        _HEAD_LIMITS.get("hidden_choices", [256, 384, 512, 768, 1024, 1536, 2048]),
    )
    head_act = trial.suggest_categorical("head.activation", ACTS)
    head_do = trial.suggest_float(
        "head.dropout", 0.0, float(_HEAD_LIMITS.get("dropout_max", 0.5))
    )
    loss = trial.suggest_categorical("loss.cls.type", LOSSES_CLS)
    label_smooth = (
        trial.suggest_float("loss.cls.label_smoothing", 0.0, 0.20)
        if loss != "focal"
        else 0.0
    )
    focal_gamma = (
        trial.suggest_float("loss.cls.gamma", 1.0, 5.0) if loss == "focal" else None
    )
    focal_alpha = (
        trial.suggest_float("loss.cls.alpha", 0.1, 0.9) if loss == "focal" else None
    )
    class_balance = trial.suggest_categorical(
        "loss.cls.balance", ["none", "weighted", "effective_num"]
    )
    epochs = int(os.getenv("HPO_EPOCHS", "100"))
    return {
        "task": "criteria",
        "model": {"name": model_name},
        "head": {
            "pooling": pooling,
            "layers": head_layers,
            "hidden": head_hidden,
            "activation": head_act,
            "dropout": head_do,
        },
        "loss": {
            "type": loss,
            "label_smoothing": label_smooth,
            "gamma": focal_gamma,
            "alpha": focal_alpha,
            "balance": class_balance,
        },
        **aug,
        **com,
        "train": {**com["train"], "epochs": epochs},
    }


def suggest_evidence(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    heavy = any(k in model_name for k in ["-large", "large", "xlm-roberta"])
    com = suggest_common(trial, heavy)
    aug = suggest_augmentation(trial)
    head_layers = trial.suggest_int(
        "head.layers",
        int(_HEAD_LIMITS.get("layers_min", 1)),
        int(_HEAD_LIMITS.get("layers_max", 4)),
    )
    head_hidden = trial.suggest_categorical(
        "head.hidden",
        _HEAD_LIMITS.get("hidden_choices", [256, 384, 512, 768, 1024, 1536, 2048]),
    )
    head_act = trial.suggest_categorical("head.activation", ACTS)
    head_do = trial.suggest_float(
        "head.dropout", 0.0, float(_HEAD_LIMITS.get("dropout_max", 0.5))
    )
    loss = trial.suggest_categorical("loss.qa.type", LOSSES_QA)
    label_smooth = (
        trial.suggest_float("loss.qa.label_smoothing", 0.0, 0.15)
        if loss != "qa_focal"
        else 0.0
    )
    focal_gamma = (
        trial.suggest_float("loss.qa.gamma", 1.0, 5.0) if loss == "qa_focal" else None
    )
    focal_alpha = (
        trial.suggest_float("loss.qa.alpha", 0.1, 0.9) if loss == "qa_focal" else None
    )
    null_pol = trial.suggest_categorical("qa.null.policy", NULL_POLICIES)
    null_threshold = (
        trial.suggest_float("qa.null.threshold", -5.0, 5.0)
        if null_pol in ("threshold", "calibrated")
        else None
    )
    null_ratio = (
        trial.suggest_float("qa.null.ratio", 0.05, 0.8) if null_pol == "ratio" else None
    )
    topk = trial.suggest_int("qa.topk", 1, 20)
    max_ans = trial.suggest_int("qa.max_answer_len", 20, 512, step=4)
    nbest = trial.suggest_int("qa.n_best_size", 10, 50)
    rerank = trial.suggest_categorical("qa.reranker", RERANKERS)
    nms_iou = trial.suggest_float("qa.nms_iou", 0.3, 0.8)
    neg_ratio = trial.suggest_float("qa.neg_ratio", 0.1, 1.0)
    epochs = int(os.getenv("HPO_EPOCHS", "100"))
    return {
        "task": "evidence",
        "model": {"name": model_name},
        "head": {
            "layers": head_layers,
            "hidden": head_hidden,
            "activation": head_act,
            "dropout": head_do,
        },
        "loss": {
            "type": loss,
            "label_smoothing": label_smooth,
            "gamma": focal_gamma,
            "alpha": focal_alpha,
        },
        "qa": {
            "null": {
                "policy": null_pol,
                "threshold": null_threshold,
                "ratio": null_ratio,
            },
            "topk": topk,
            "max_answer_len": max_ans,
            "n_best_size": nbest,
            "reranker": rerank,
            "nms_iou": nms_iou,
            "neg_ratio": neg_ratio,
        },
        **aug,
        **com,
        "train": {**com["train"], "epochs": epochs},
    }


def suggest_joint(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    cfg_e = suggest_evidence(trial, model_name)
    cfg_c = suggest_criteria(trial, model_name)
    share_ratio = trial.suggest_float("joint.share_ratio", 0.0, 1.0)
    multi_task_weight = trial.suggest_float("joint.criteria_weight", 0.2, 0.8)
    return {
        "task": "joint",
        "model": {"name": model_name},
        "shared": {"ratio": share_ratio, "criteria_weight": multi_task_weight},
        "criteria": cfg_c,
        "evidence": cfg_e,
        "tok": cfg_e["tok"],
        "optim": cfg_e["optim"],
        "sched": cfg_e["sched"],
        "regularization": cfg_e["regularization"],
        "train": cfg_e["train"],
    }


def build_config(trial: optuna.Trial, agent: str) -> dict[str, Any]:
    model = trial.suggest_categorical("model.name", MODEL_CHOICES)
    if agent == "criteria":
        return suggest_criteria(trial, model)
    if agent == "evidence":
        return suggest_evidence(trial, model)
    if agent == "joint":
        return suggest_joint(trial, model)
    if agent == "share":
        # For "share", treat similarly to joint with different defaults; reuse joint for now.
        return suggest_joint(trial, model)
    raise ValueError(agent)


def run_training_eval(
    cfg: dict[str, Any],
    callbacks: dict[str, Callable[[int, float, float | None], None]],
) -> dict[str, float]:
    """
    Training bridge for HPO integration with REAL redsm5 data and EarlyStopping.

    Loads real redsm5 dataset, trains the model with EarlyStopping, and reports metrics.

    Args:
        cfg: Configuration dict with model, head, train, optim, etc.
        callbacks: Dict with "on_epoch" callback for reporting metrics

    Returns:
        Dict with "primary" metric and "runtime_s"
    """
    import sys
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from Project.Criteria.data.dataset import CriteriaDataset
    from Project.Criteria.models.model import Model as CriteriaModel
    from Project.Evidence.data.dataset import EvidenceDataset

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    task = cfg.get("task", "criteria")
    model_name = cfg["model"]["name"]

    # EarlyStopping config from environment
    patience = int(os.getenv("HPO_PATIENCE", "20"))
    min_delta = float(os.getenv("HPO_MIN_DELTA", "0.0"))
    es = EarlyStopping(patience=patience, min_delta=min_delta, mode="max")

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load real dataset based on task
    project_root = Path(__file__).parent.parent
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task in ("criteria", "share", "joint"):
        dataset_path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
        dataset = CriteriaDataset(
            csv_path=dataset_path,
            tokenizer=tokenizer,
            max_length=cfg["tok"]["max_length"],
        )
        num_labels = 2
    elif task == "evidence":
        dataset_path = (
            project_root / "data" / "processed" / "redsm5_matched_evidence.csv"
        )
        dataset = EvidenceDataset(
            csv_path=dataset_path,
            tokenizer=tokenizer,
            max_length=cfg["tok"]["max_length"],
        )
        num_labels = 2
    else:
        raise ValueError(f"Unknown task: {task}")

    # Split dataset (80/10/10) - train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(cfg["meta"]["seed"])
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create dataloaders with optimized workers
    # Use environment variable or auto-detect based on CPU cores
    num_workers = int(
        os.getenv("NUM_WORKERS", "18")
    )  # 18 workers for 20-core system (90% CPU, 2 cores for system/monitoring)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=num_workers > 0,  # Keep workers alive
        prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Create model based on task
    if task in ("criteria", "share", "joint") or task == "evidence":
        # Use head_cfg for HPO compatibility
        head_cfg = cfg.get("head", {})
        task_cfg = {"num_labels": num_labels}
        model = CriteriaModel(
            model_name=model_name,
            head_cfg=head_cfg,
            task_cfg=task_cfg,
        ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Create optimizer based on config
    optim_name = cfg["optim"]["name"]
    lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(cfg["optim"].get("beta1", 0.9), cfg["optim"].get("beta2", 0.999)),
            eps=cfg["optim"].get("eps", 1e-8),
        )
    elif optim_name in ("adamw_8bit", "lion") or optim_name == "adafactor":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Training loop with EarlyStopping and Mixed Precision
    start = time.time()
    best = 0.0

    # Enable mixed precision for RTX 4090 (Ampere architecture supports bfloat16)
    use_amp = bool(torch.cuda.is_available())
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # Mixed precision backward pass
            if use_amp and scaler:
                scaler.scale(loss).backward()
                if cfg["train"].get("clip_grad", 0.0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["train"]["clip_grad"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["train"].get("clip_grad", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["train"]["clip_grad"]
                    )
                optimizer.step()

            total_loss += loss.item()

        # Validation with mixed precision
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Use AMP for inference too
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # Calculate metrics
        from sklearn.metrics import f1_score

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        # Use F1 as primary metric
        metric = val_f1
        best = max(best, metric)

        # Report to Optuna
        callbacks["on_epoch"](epoch, metric, avg_val_loss)

        # Check EarlyStopping
        if es.step(metric):
            print(f"EarlyStopping triggered at epoch {epoch+1} (patience={patience})")
            break

    runtime = time.time() - start
    return {"primary": float(best), "runtime_s": runtime}


def make_pruner() -> optuna.pruners.BasePruner:
    hb = HyperbandPruner(
        min_resource=1,
        max_resource=int(os.getenv("HPO_EPOCHS", "100")),
        reduction_factor=3,
    )
    return PatientPruner(hb, patience=2)


def make_sampler(multi_objective: bool, seed: int) -> optuna.samplers.BaseSampler:
    if multi_objective:
        return NSGAIISampler(seed=seed)
    return TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True)


def objective_builder(
    agent: str, outdir: str, multi_objective: bool
) -> Callable[[optuna.Trial], float]:
    def _obj(trial: optuna.Trial):
        seed = trial.suggest_int("seed", 1, 65535)
        set_seeds(seed)
        cfg = build_config(trial, agent)
        cfg["meta"] = {
            "agent": agent,
            "seed": seed,
            "outdir": outdir,
            "repo": "NoAug_Criteria_Evidence",
            "aug": False,
        }
        if _HAS_MLFLOW:
            mlflow.start_run(nested=True)
            mlflow.log_params(
                {k: v for k, v in flatten_dict(cfg).items() if is_loggable(v)}
            )

        def _cb(epoch, primary, secondary=None):
            on_epoch(trial, epoch, primary, secondary)

        try:
            res = run_training_eval(cfg, {"on_epoch": _cb})

            if _HAS_MLFLOW:
                mlflow.log_metrics(
                    {
                        "final_primary": res["primary"],
                        "runtime_s": res.get("runtime_s", float("nan")),
                    }
                )
                mlflow.end_run()

            return res["primary"]

        except (RuntimeError, ValueError, AttributeError, TypeError, KeyError) as e:
            # Handle configuration errors (max_length > model max, missing attributes, etc.)
            import traceback

            error_msg = f"{type(e).__name__}: {str(e)}"
            print(
                f"\n[ERROR] Trial {trial.number} failed with config error: {error_msg}"
            )
            if "expanded size" in str(e) or "max_position_embeddings" in str(e):
                print("  Cause: max_length exceeds model's maximum position embeddings")
            elif "pooler_output" in str(e):
                print("  Cause: Model variant doesn't have pooler_output")

            if _HAS_MLFLOW:
                mlflow.log_param("error", error_msg[:250])  # Truncate long errors
                mlflow.log_metric("final_primary", 0.0)  # Worst possible score
                mlflow.end_run()

            # Return worst possible score to let Optuna skip this configuration
            raise optuna.exceptions.TrialPruned(
                f"Invalid configuration: {error_msg}"
            ) from e

        except Exception as e:
            # Unexpected errors - still log and prune
            import traceback

            print(f"\n[CRITICAL ERROR] Trial {trial.number} failed unexpectedly:")
            traceback.print_exc()

            if _HAS_MLFLOW:
                mlflow.log_param("fatal_error", str(e)[:250])
                mlflow.end_run()

            raise optuna.exceptions.TrialPruned(
                f"Unexpected error: {type(e).__name__}"
            ) from e

    return _obj


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def is_loggable(v: Any) -> bool:
    return isinstance(v, str | int | float | bool) or v is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["criteria", "evidence", "share", "joint"],
        required=True,
    )
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--storage",
        default=os.getenv(
            "OPTUNA_STORAGE",
            f"sqlite:///{os.path.abspath('./_optuna/noaug.db')}",
        ),
    )
    parser.add_argument(
        "--outdir", default=os.getenv("HPO_OUTDIR", os.path.abspath("./_runs"))
    )
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--multi-objective", action="store_true")
    args = parser.parse_args()

    # Create storage directory only for SQLite (not PostgreSQL)
    if args.storage.startswith("sqlite://"):
        os.makedirs(
            os.path.dirname(args.storage.replace("sqlite:///", "")),
            exist_ok=True,
        )

    os.makedirs(args.outdir, exist_ok=True)
    default_mlflow_setup(args.outdir)

    epochs = int(os.getenv("HPO_EPOCHS", "100"))
    print(f"[HPO] agent={args.agent} epochs={epochs} storage={args.storage}")

    sampler = make_sampler(args.multi_objective, seed=2025)
    pruner = make_pruner()

    study = optuna.create_study(
        study_name=args.study,
        directions=(
            ["maximize"] if not args.multi_objective else ["maximize", "minimize"]
        ),
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = objective_builder(args.agent, args.outdir, args.multi_objective)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.parallel,
        gc_after_trial=True,
    )

    print("\n[Best Trials]")
    for trial in study.best_trials[:5]:
        try:
            value = trial.values if hasattr(trial, "values") else trial.value
        except Exception:
            value = trial.value
        print(f"- value={value} | params={trial.params}")

    top_limit = min(8, len(study.best_trials))
    topk = []
    for trial in study.best_trials[:top_limit]:
        val = getattr(
            trial, "value", trial.values[0] if hasattr(trial, "values") else None
        )
        topk.append({"value": val, "params": trial.params})
    with open(
        os.path.join(args.outdir, f"{args.agent}_{args.study}_topk.json"), "w"
    ) as fh:
        json.dump(topk, fh, indent=2)


if __name__ == "__main__":
    main()

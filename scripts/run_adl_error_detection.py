"""
Task-execution error detection on CASAS Kyoto ADL (episode level)
=================================================================
Compares, under participant-grouped stratified k-fold CV:
  - Rules       : procedural rules over task models (no error examples used)
  - LogReg / RF : classical models on bag-of-sensor summary features
  - GRU         : CasasGRUClassifier over the episode event sequence
  - GRU-motion  : GRU on motion events only (confound control)
  - NeSy        : Rules OR GRU, with rule-grounded explanations

Outputs (under adl_errors.output_dir):
  predictions.csv, summary.json, per_task.csv, explanations.csv,
  confound.json, tables/*.tex

Usage (Bash):
  python -m scripts.run_adl_error_detection --config config/adl_error_detection.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from adl_nesy.data.episodes import (
    EpisodeSet,
    build_episodes,
    episode_features,
    episode_token_sequences,
    final_analog_readings,
)
from adl_nesy.neural.gru import CasasGRUClassifier
from adl_nesy.neural.trainer import train_loop
from adl_nesy.neural.utils import ensure_dir, pick_device, save_json
from adl_nesy.reasoning.procedural_rules import (
    calibrate_thresholds,
    detect_errors,
    load_task_models,
)
from adl_nesy.utils.config import load_config
from adl_nesy.utils.logging import setup_logging
from adl_nesy.utils.seed import set_global_seed

log = logging.getLogger(__name__)

ID2LABEL = ["Correct", "Error"]

# Explanation expected for each injected error (Zenodo protocol). PhoneCall's
# error (wrong number dialled) has no sensor that could evidence it.
EXPECTED_EXPLANATION = {
    "Clean": "Step_RinseDishes",
    "Eat": "Step_TakeMedicine",
    "Cook": "Post_BurnerOff",
    "WashHands": "Post_WaterOff",
    "PhoneCall": None,
}
TASK_ORDER = ["PhoneCall", "WashHands", "Cook", "Eat", "Clean"]
METHODS = ["Rules", "IForest", "LogReg", "RF", "GRU", "GRU-motion", "NeSy"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ADL task-execution error detection (episode level).")
    ap.add_argument("--config", type=str, default="config/adl_error_detection.yaml")
    ap.add_argument("--out", type=str, default=None, help="Override adl_errors.output_dir")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated neural seeds (overrides config)")
    ap.add_argument("--scarcity", action="store_true", help="Also run the label-scarcity sweep")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Neural sequence data
# ---------------------------------------------------------------------------
class EpisodeSequenceDataset(Dataset):
    def __init__(self, token_ids: List[List[int]], deltas: List[List[float]], labels: List[int], max_len: int, time_scale_s: float):
        self.token_ids = [t[:max_len] for t in token_ids]
        self.deltas = [[x / time_scale_s for x in d[:max_len]] for d in deltas]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[float], int, int]:
        return self.token_ids[idx], self.deltas[idx], len(self.token_ids[idx]), self.labels[idx]


def episode_collate(batch: List[Tuple[List[int], List[float], int, int]]) -> Tuple[torch.Tensor, ...]:
    """Pad to batch max; keeps absolute (scaled) time so episode duration is visible."""
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tok = torch.zeros((len(batch), max_len), dtype=torch.long)
    td = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, (t, d, n, _) in enumerate(batch):
        tok[i, :n] = torch.tensor(t, dtype=torch.long)
        td[i, :n] = torch.tensor(d, dtype=torch.float32)
    y = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return tok, td, lengths, y


def _encode(seqs: Dict[str, Tuple[List[str], List[float]]], ids: List[str], vocab: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
    toks = [[vocab.get(t, 1) for t in seqs[i][0]] for i in ids]
    dts = [seqs[i][1] for i in ids]
    return toks, dts


@torch.no_grad()
def _predict_proba(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    for tok, td, lengths, _ in loader:
        logits = model(tok.to(device), td.to(device), lengths.to(device))
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_gru_fold(
    seqs: Dict[str, Tuple[List[str], List[float]]],
    episodes: pd.DataFrame,
    train_ids: List[str],
    test_ids: List[str],
    gcfg: Dict[str, Any],
    seed: int,
    device: str,
    out_dir: Path,
) -> Tuple[np.ndarray, float]:
    """Train a GRU on train_ids (inner participant-grouped validation split);
    return P(error) for test_ids and mean inference seconds per episode."""
    set_global_seed(seed)
    # set_global_seed covers Python and NumPy only; seed PyTorch so that
    # weight initialisation and dropout are reproducible.
    torch.manual_seed(seed)
    ep = episodes.set_index("episode_id")

    # Inner validation split: one fold of a grouped stratified split
    tr = ep.loc[train_ids]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(inner.split(tr.index, tr["has_error"].astype(int), groups=tr["participant"]))
    fit_ids = [train_ids[i] for i in fit_idx]
    val_ids = [train_ids[i] for i in val_idx]

    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for i in fit_ids:
        for t in seqs[i][0]:
            vocab.setdefault(t, len(vocab))

    def loader(ids: List[str], shuffle: bool) -> DataLoader:
        toks, dts = _encode(seqs, ids, vocab)
        labels = [int(ep.at[i, "has_error"]) for i in ids]
        ds = EpisodeSequenceDataset(toks, dts, labels, int(gcfg["max_seq_len"]), float(gcfg["time_scale_s"]))
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, batch_size=int(gcfg["batch_size"]), shuffle=shuffle, collate_fn=episode_collate, generator=g)

    model = CasasGRUClassifier(
        vocab_size=len(vocab),
        num_classes=2,
        emb_dim=int(gcfg["emb_dim"]),
        hidden=int(gcfg["hidden_size"]),
        num_layers=int(gcfg["num_layers"]),
        dropout=float(gcfg["dropout"]),
    )
    test_loader = loader(test_ids, False)
    train_loop(
        model=model,
        train_loader=loader(fit_ids, True),
        val_loader=loader(val_ids, False),
        test_loader=test_loader,
        device=device,
        id2label=ID2LABEL,
        out_dir=out_dir,
        epochs=int(gcfg["epochs"]),
        lr=float(gcfg["lr"]),
        seed=seed,
    )
    t0 = time.perf_counter()
    proba = _predict_proba(model, test_loader, device)
    per_ep = (time.perf_counter() - t0) / max(len(test_ids), 1)
    return proba, per_ep


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None) -> Dict[str, float]:
    m = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "false_alarms": int(((pred == 1) & (y == 0)).sum()),
        "missed": int(((pred == 0) & (y == 1)).sum()),
        "n": int(len(y)),
    }
    if score is not None and len(np.unique(y)) == 2:
        m["auroc"] = float(roc_auc_score(y, score))
    return m


def cluster_bootstrap(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    n_resamples: int,
    seed: int,
) -> Tuple[float, float]:
    """95% percentile CI, resampling participants (the independent unit)."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("participant")}
    keys = np.array(list(groups.keys()))
    vals = []
    for _ in range(n_resamples):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _f1_of(method: str) -> Callable[[pd.DataFrame], float]:
    return lambda d: float(f1_score(d["y"], d[f"pred_{method}"], zero_division=0))


def _f1_diff(a: str, b: str) -> Callable[[pd.DataFrame], float]:
    return lambda d: _f1_of(a)(d) - _f1_of(b)(d)


# ---------------------------------------------------------------------------
# Confound check: idle analog levels and recording periods by cohort
# ---------------------------------------------------------------------------
def confound_report(eps: EpisodeSet) -> Dict[str, Any]:
    ev = eps.events.dropna(subset=["value_num"])
    first = ev.groupby(["episode_id", "sensor"])["value_num"].first().reset_index()
    first = first.merge(eps.episodes[["episode_id", "has_error", "task_name"]], on="episode_id")
    out: Dict[str, Any] = {"idle_level_first_reading": {}}
    for sensor, task in [("AD1-A", "Cook"), ("AD1-B", None), ("AD1-C", None)]:
        sub = first[first["sensor"] == sensor]
        if task:
            sub = sub[sub["task_name"] == task]
        a = sub.loc[sub["has_error"], "value_num"]
        b = sub.loc[~sub["has_error"], "value_num"]
        if len(a) and len(b):
            out["idle_level_first_reading"][sensor] = {
                "error_median": float(a.median()),
                "correct_median": float(b.median()),
                "mannwhitney_p": float(mannwhitneyu(a, b).pvalue),
                "n_error": int(len(a)),
                "n_correct": int(len(b)),
            }
    months = eps.episodes.assign(month=eps.episodes["start_time"].dt.strftime("%Y-%m"))
    out["episodes_by_month"] = (
        months.groupby(["month", "has_error"]).size().unstack(fill_value=0).rename(columns={False: "correct", True: "error"}).to_dict(orient="index")
    )
    return out


# ---------------------------------------------------------------------------
# LaTeX tables (numbers are written straight from the results)
# ---------------------------------------------------------------------------
def write_tables(summary: Dict[str, Any], per_task: pd.DataFrame, tables_dir: Path) -> None:
    ensure_dir(tables_dir)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Precision & Recall & F1 [95\% CI] & Bal.\ acc. & AUROC & False alarms \\",
        r"\midrule",
    ]
    for m in METHODS:
        s = summary["pooled"][m]
        lo, hi = summary["f1_ci"][m]
        auroc = f"{s['auroc']:.3f}" if "auroc" in s else "--"
        lines.append(
            f"{m} & {s['precision']:.3f} & {s['recall']:.3f} & {s['f1']:.3f} [{lo:.2f}, {hi:.2f}] & "
            f"{s['balanced_accuracy']:.3f} & {auroc} & {s['false_alarms']}/{summary['n_correct']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables_dir / "overall.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = ["Rules", "LogReg", "RF", "GRU", "NeSy"]
    lines = [
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Task (injected error) & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    names = {
        "PhoneCall": "PhoneCall (wrong number)",
        "WashHands": "WashHands (water left on)",
        "Cook": "Cook (burner left on)",
        "Eat": "Eat (no medicine)",
        "Clean": "Clean (no water)",
    }
    for task in TASK_ORDER:
        row = per_task[per_task["task"] == task].set_index("method")
        vals = [f"{row.at[c, 'f1']:.2f}" for c in cols]
        best = max(float(v) for v in vals)
        vals = [f"\\textbf{{{v}}}" if float(v) == best else v for v in vals]
        lines.append(f"{names[task]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables_dir / "per_task_f1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Label scarcity: supervised models with few error-labelled participants
# ---------------------------------------------------------------------------
def run_scarcity(
    *,
    episodes: pd.DataFrame,
    feats: pd.DataFrame,
    seqs: Dict[str, Tuple[List[str], List[float]]],
    preds: pd.DataFrame,
    folds: List[Tuple[List[str], List[str]]],
    levels: List[int],
    reps: int,
    gcfg: Dict[str, Any],
    thr: float,
    seed0: int,
    device: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Train with all correct executions but only ``m`` error-labelled
    participants per fold. Rules are unaffected (they use no error labels)."""
    ep = episodes.set_index("episode_id")
    rows = []
    for m in levels:
        for rep in range(reps):
            rng = np.random.default_rng(seed0 + 1000 * m + rep)
            pooled: Dict[str, List[np.ndarray]] = {"y": [], "LogReg": [], "RF": [], "GRU": [], "NeSy": [], "Rules": [], "IForest": []}
            for fold, (tr_ids, te_ids) in enumerate(folds):
                tr = ep.loc[tr_ids]
                err_people = sorted(tr.loc[tr["has_error"], "participant"].unique())
                keep_err = set(rng.choice(err_people, size=min(m, len(err_people)), replace=False))
                sub_ids = [i for i in tr_ids if (not ep.at[i, "has_error"]) or ep.at[i, "participant"] in keep_err]
                ysub = ep.loc[sub_ids, "has_error"].astype(int).to_numpy()
                yte = ep.loc[te_ids, "has_error"].astype(int).to_numpy()

                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed0))
                lr.fit(feats.loc[sub_ids].to_numpy(), ysub)
                rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed0, n_jobs=-1)
                rf.fit(feats.loc[sub_ids].to_numpy(), ysub)
                p_gru, _ = train_gru_fold(seqs, episodes, sub_ids, te_ids, gcfg, seed0 + rep, device, out_dir / "scarcity" / f"m{m}_r{rep}_f{fold}")
                rules = preds.set_index("episode_id").loc[te_ids, "pred_Rules"].to_numpy()
                iforest = preds.set_index("episode_id").loc[te_ids, "pred_IForest"].to_numpy()

                pooled["y"].append(yte)
                pooled["LogReg"].append((lr.predict_proba(feats.loc[te_ids].to_numpy())[:, 1] >= thr).astype(int))
                pooled["RF"].append((rf.predict_proba(feats.loc[te_ids].to_numpy())[:, 1] >= thr).astype(int))
                pooled["GRU"].append((p_gru >= thr).astype(int))
                pooled["NeSy"].append(((p_gru >= thr) | (rules == 1)).astype(int))
                pooled["Rules"].append(rules)
                pooled["IForest"].append(iforest)
            y = np.concatenate(pooled["y"])
            for meth in ["LogReg", "RF", "GRU", "NeSy", "Rules", "IForest"]:
                p = np.concatenate(pooled[meth])
                rows.append({"error_participants": m, "rep": rep, "method": meth, **binary_metrics(y, p, None)})
            log.info("Scarcity m=%d rep=%d done", m, rep)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging()
    acfg = cfg.get("adl_errors", {})
    seed0 = int(cfg.get("project", {}).get("seed", 42))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in acfg.get("seeds", [seed0])]
    out_dir = Path(args.out or acfg.get("output_dir", "outputs/adl_errors"))
    ensure_dir(out_dir)
    set_global_seed(seed0)
    device = pick_device(cfg.get("neural_perception", {}).get("device", "auto"))
    gcfg = acfg.get("gru", {})
    thr = float(acfg.get("decision_threshold", 0.5))

    eps = build_episodes(cfg)
    episodes = eps.episodes
    finals = final_analog_readings(eps)
    feats = episode_features(eps).loc[episodes["episode_id"]]
    task_models = load_task_models()
    levels = acfg.get("levels", {})
    seqs_full = episode_token_sequences(eps, levels)
    seqs_motion = episode_token_sequences(eps, levels, sensor_prefixes=("M",))

    y_all = episodes["has_error"].astype(int).to_numpy()
    groups = episodes["participant"].to_numpy()
    ids_all = episodes["episode_id"].tolist()
    k = int(acfg.get("k_folds", 5))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed0)

    preds = episodes[["episode_id", "participant", "task_name", "has_error"]].copy()
    preds["y"] = y_all
    preds["fold"] = -1
    for col in ["score_Rules", "score_IForest", "pred_IForest", "score_LogReg", "score_RF"]:
        preds[col] = np.nan
    for s in seeds:
        preds[f"score_GRU_s{s}"] = np.nan
        preds[f"score_GRU-motion_s{s}"] = np.nan
    preds["missing_steps"] = ""
    preds["violated_postconditions"] = ""

    latency: Dict[str, List[float]] = {"Rules": [], "GRU": [], "LogReg": []}
    thresholds_per_fold: List[Dict[str, float]] = []
    folds: List[Tuple[List[str], List[str]]] = []
    leakage_ok = True

    for fold, (tr_idx, te_idx) in enumerate(splitter.split(ids_all, y_all, groups=groups)):
        tr_ids = [ids_all[i] for i in tr_idx]
        te_ids = [ids_all[i] for i in te_idx]
        folds.append((tr_ids, te_ids))
        te_mask = preds["episode_id"].isin(te_ids)
        preds.loc[te_mask, "fold"] = fold
        if set(groups[tr_idx]) & set(groups[te_idx]):
            leakage_ok = False
        log.info("Fold %d: train=%d test=%d (test errors=%d)", fold, len(tr_ids), len(te_ids), int(y_all[te_idx].sum()))

        # Rules: thresholds from correct training executions only
        th = calibrate_thresholds(task_models, episodes[episodes["episode_id"].isin(tr_ids)], finals)
        thresholds_per_fold.append(th)
        res, timing = detect_errors(eps, te_ids, finals, th, task_models)
        latency["Rules"].append((timing["kg_build_s"] + timing["reasoning_s"]) / len(te_ids))
        res = res.set_index("episode_id").loc[preds.loc[te_mask, "episode_id"]]
        preds.loc[te_mask, "score_Rules"] = res["rule_error"].astype(float).to_numpy()
        preds.loc[te_mask, "missing_steps"] = res["missing_steps"].to_numpy()
        preds.loc[te_mask, "violated_postconditions"] = res["violated_postconditions"].to_numpy()

        # Classical baselines
        Xtr, Xte = feats.loc[tr_ids].to_numpy(), feats.loc[preds.loc[te_mask, "episode_id"]].to_numpy()
        ytr = y_all[tr_idx]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed0))
        lr.fit(Xtr, ytr)
        t0 = time.perf_counter()
        preds.loc[te_mask, "score_LogReg"] = lr.predict_proba(Xte)[:, 1]
        latency["LogReg"].append((time.perf_counter() - t0) / len(te_ids))
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed0, n_jobs=-1)
        rf.fit(Xtr, ytr)
        preds.loc[te_mask, "score_RF"] = rf.predict_proba(Xte)[:, 1]

        # Label-free data-driven baseline: per-task Isolation Forest fitted on
        # correct training executions; threshold = least normal training score
        # (the same calibration principle as the rules' off-levels).
        num_cols = [c for c in feats.columns if not c.startswith("task_")]
        for task in TASK_ORDER:
            tr_ok = episodes[episodes["episode_id"].isin(tr_ids) & (episodes["task_name"] == task) & ~episodes["has_error"]]["episode_id"]
            te_t = preds.loc[te_mask & (preds["task_name"] == task), "episode_id"]
            iso = IsolationForest(n_estimators=500, random_state=seed0)
            iso.fit(feats.loc[tr_ok, num_cols].to_numpy())
            thr_iso = float(iso.score_samples(feats.loc[tr_ok, num_cols].to_numpy()).min())
            s_te = iso.score_samples(feats.loc[te_t, num_cols].to_numpy())
            rows = preds["episode_id"].isin(te_t)
            preds.loc[rows, "score_IForest"] = -s_te
            preds.loc[rows, "pred_IForest"] = (s_te < thr_iso).astype(int)

        # GRU models (ids in the preds row order of this fold)
        te_order = preds.loc[te_mask, "episode_id"].tolist()
        for s in seeds:
            p, per_ep = train_gru_fold(seqs_full, episodes, tr_ids, te_order, gcfg, s, device, out_dir / "gru" / f"seed{s}" / f"fold{fold}")
            preds.loc[te_mask, f"score_GRU_s{s}"] = p
            if s == seeds[0]:
                latency["GRU"].append(per_ep)
            p, _ = train_gru_fold(seqs_motion, episodes, tr_ids, te_order, gcfg, s, device, out_dir / "gru_motion" / f"seed{s}" / f"fold{fold}")
            preds.loc[te_mask, f"score_GRU-motion_s{s}"] = p

    # Reference seed drives the main tables; other seeds give variability
    ref = seeds[0]
    preds["score_GRU"] = preds[f"score_GRU_s{ref}"]
    preds["score_GRU-motion"] = preds[f"score_GRU-motion_s{ref}"]
    preds["pred_Rules"] = preds["score_Rules"].astype(int)
    preds["pred_IForest"] = preds["pred_IForest"].astype(int)
    for m in ["LogReg", "RF", "GRU", "GRU-motion"]:
        preds[f"pred_{m}"] = (preds[f"score_{m}"] >= thr).astype(int)
    preds["pred_NeSy"] = ((preds["pred_Rules"] == 1) | (preds["pred_GRU"] == 1)).astype(int)
    preds["score_NeSy"] = np.maximum(preds["score_Rules"], preds["score_GRU"])
    preds.to_csv(out_dir / "predictions.csv", index=False)

    y = preds["y"].to_numpy()
    n_boot = int(acfg.get("bootstrap_resamples", 2000))
    summary: Dict[str, Any] = {
        "n_episodes": int(len(preds)),
        "n_error": int(y.sum()),
        "n_correct": int((1 - y).sum()),
        "n_participants": int(preds["participant"].nunique()),
        "k_folds": k,
        "seeds": seeds,
        "decision_threshold": thr,
        "participant_leakage_free": leakage_ok,
        "pooled": {},
        "per_fold_f1": {},
        "f1_ci": {},
        "paired_f1_diff_ci": {},
        "thresholds_per_fold": thresholds_per_fold,
        "latency_ms_per_episode": {m: 1000 * float(np.mean(v)) for m, v in latency.items() if v},
        "trivial_all_error": binary_metrics(y, np.ones_like(y), None),
    }
    for m in METHODS:
        score = preds[f"score_{m}"].to_numpy()
        summary["pooled"][m] = binary_metrics(y, preds[f"pred_{m}"].to_numpy(), score)
        summary["per_fold_f1"][m] = [
            float(f1_score(g["y"], g[f"pred_{m}"], zero_division=0)) for _, g in preds.groupby("fold")
        ]
        summary["f1_ci"][m] = cluster_bootstrap(preds, _f1_of(m), n_boot, seed0)
    for a, b in [("NeSy", "GRU"), ("NeSy", "Rules"), ("NeSy", "RF"), ("NeSy", "LogReg"), ("Rules", "IForest"), ("Rules", "RF")]:
        summary["paired_f1_diff_ci"][f"{a}-{b}"] = {
            "diff": _f1_of(a)(preds) - _f1_of(b)(preds),
            "ci95": cluster_bootstrap(preds, _f1_diff(a, b), n_boot, seed0),
        }
    if len(seeds) > 1:
        seed_f1: Dict[str, List[float]] = {"GRU": [], "NeSy": [], "GRU-motion": []}
        for s in seeds:
            pg = (preds[f"score_GRU_s{s}"] >= thr).astype(int)
            pm = (preds[f"score_GRU-motion_s{s}"] >= thr).astype(int)
            seed_f1["GRU"].append(float(f1_score(y, pg)))
            seed_f1["GRU-motion"].append(float(f1_score(y, pm)))
            seed_f1["NeSy"].append(float(f1_score(y, ((preds["pred_Rules"] == 1) | (pg == 1)).astype(int))))
        summary["seed_f1"] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for m, v in seed_f1.items()}

    # Per-task metrics
    rows = []
    for task in TASK_ORDER:
        sub = preds[preds["task_name"] == task]
        for m in METHODS:
            met = binary_metrics(sub["y"].to_numpy(), sub[f"pred_{m}"].to_numpy(), sub[f"score_{m}"].to_numpy())
            rows.append({"task": task, "method": m, **met})
    per_task = pd.DataFrame(rows)
    per_task.to_csv(out_dir / "per_task.csv", index=False)

    # Explanations: coverage and fidelity of rule-grounded reasons
    expl = preds.copy()
    expl["explanation"] = (expl["missing_steps"] + ";" + expl["violated_postconditions"]).str.strip(";")
    tp_nesy = expl[(expl["pred_NeSy"] == 1) & (expl["y"] == 1)]
    tp_rules = expl[(expl["pred_Rules"] == 1) & (expl["y"] == 1)]

    def _faithful(r: pd.Series) -> bool:
        want = EXPECTED_EXPLANATION.get(r["task_name"])
        return bool(want) and want in r["explanation"].split(";")

    summary["explanations"] = {
        "nesy_true_detections": int(len(tp_nesy)),
        "nesy_true_detections_with_rule_reason": int((tp_nesy["pred_Rules"] == 1).sum()),
        "rule_true_detections": int(len(tp_rules)),
        "rule_true_detections_matching_injected_error": int(tp_rules.apply(_faithful, axis=1).sum()),
        "rule_false_alarm_reasons": expl[(expl["pred_Rules"] == 1) & (expl["y"] == 0)]["explanation"].value_counts().to_dict(),
        "rule_mismatched_reasons_on_errors": tp_rules[~tp_rules.apply(_faithful, axis=1)]["explanation"].value_counts().to_dict(),
    }
    expl[["episode_id", "task_name", "has_error", "pred_Rules", "pred_GRU", "pred_NeSy", "score_GRU", "explanation"]].to_csv(
        out_dir / "explanations.csv", index=False
    )

    save_json(out_dir / "confound.json", confound_report(eps))
    save_json(out_dir / "summary.json", summary)
    write_tables(summary, per_task, out_dir / "tables")

    for m in METHODS:
        s = summary["pooled"][m]
        log.info("%-10s P=%.3f R=%.3f F1=%.3f CI=[%.3f, %.3f] FA=%d", m, s["precision"], s["recall"], s["f1"], *summary["f1_ci"][m], s["false_alarms"])

    if args.scarcity:
        sc = acfg.get("scarcity", {})
        scar = run_scarcity(
            episodes=episodes,
            feats=feats,
            seqs=seqs_full,
            preds=preds,
            folds=folds,
            levels=[int(x) for x in sc.get("levels", [1, 2, 4, 8])],
            reps=int(sc.get("reps", 3)),
            gcfg=gcfg,
            thr=thr,
            seed0=seed0,
            device=device,
            out_dir=out_dir,
        )
        scar.to_csv(out_dir / "scarcity.csv", index=False)
        log.info("\n%s", scar.groupby(["error_participants", "method"])["f1"].agg(["mean", "std"]).round(3).to_string())
    log.info("Results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

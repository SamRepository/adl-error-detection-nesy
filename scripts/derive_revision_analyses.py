"""
Derived analyses for the revised paper (no model retraining)
============================================================
Reads the original outputs of run_adl_error_detection.py and the raw corpus
(both read-only) and writes all additional numbers used by the revision to a
separate directory:

  confusion.json         TP/FP/FN/TN, precision, recall, F1, false-positive rate
  fusion_tradeoff.json   alerts that OR fusion adds to the rules
  explanations.json      expected-error reason inclusion and alert coverage
  seed_variation.json    run-to-run variation of GRU-based models (5 seeds)
  first_readings.json    participant-level first recorded analog readings
                         (descriptive only; no hypothesis tests)
  iforest_cutoffs.json   post-hoc isolation-forest cutoff sensitivity
  rule_rerun.json        strict-mode rule rerun compared with saved predictions
  manifest.json          versions, hardware, data/output checksums, fold
                         assignments, code revision

Usage:
  python -m scripts.derive_revision_analyses \
      --config config/adl_error_detection.yaml \
      --results <original outputs/adl_errors> --out <revision results dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score, f1_score

from adl_nesy.data.episodes import build_episodes, episode_features, final_analog_readings
from adl_nesy.reasoning.procedural_rules import calibrate_thresholds, detect_errors, load_task_models
from adl_nesy.utils.config import load_config
from adl_nesy.utils.logging import setup_logging

log = logging.getLogger(__name__)

METHODS = ["Rules", "IForest", "LogReg", "RF", "GRU", "GRU-motion", "NeSy"]
TASK_ORDER = ["PhoneCall", "WashHands", "Cook", "Eat", "Clean"]
EXPECTED_REASON = {
    "Clean": "Step_RinseDishes",
    "Eat": "Step_TakeMedicine",
    "Cook": "Post_BurnerOff",
    "WashHands": "Post_WaterOff",
    "PhoneCall": None,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Derived analyses for the revised paper.")
    ap.add_argument("--config", type=str, default="config/adl_error_detection.yaml")
    ap.add_argument("--results", type=str, required=True, help="Original outputs/adl_errors directory (read-only)")
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()


def _confusion(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "alerts": tp + fp,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reasons(row: pd.Series) -> List[str]:
    parts = [str(row["missing_steps"]), str(row["violated_postconditions"])]
    return [r for p in parts for r in p.split(";") if r and r != "nan"]


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging()
    res = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed0 = int(cfg.get("project", {}).get("seed", 42))

    preds = pd.read_csv(res / "predictions.csv")
    summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    y = preds["y"].to_numpy()

    # 1. Confusion counts for every method
    confusion = {m: _confusion(y, preds[f"pred_{m}"].to_numpy()) for m in METHODS}
    confusion["trivial_all_error"] = _confusion(y, np.ones_like(y))
    (out / "confusion.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    # 2. What OR fusion adds to the rules
    added = preds[(preds["pred_NeSy"] == 1) & (preds["pred_Rules"] == 0)]
    fusion = {
        "added_true_detections": int(added["y"].sum()),
        "added_false_alarms": int((added["y"] == 0).sum()),
        "added_alert_precision": float(added["y"].mean()) if len(added) else 0.0,
        "rules": confusion["Rules"],
        "fusion": confusion["NeSy"],
        "fusion_equals_rules_or_gru": bool(
            (preds["pred_NeSy"] == ((preds["pred_Rules"] == 1) | (preds["pred_GRU"] == 1)).astype(int)).all()
        ),
    }
    (out / "fusion_tradeoff.json").write_text(json.dumps(fusion, indent=2), encoding="utf-8")

    # 3. Explanations: expected-error reason inclusion and coverage
    rule_tp = preds[(preds["pred_Rules"] == 1) & (preds["y"] == 1)].copy()
    rule_tp["reasons"] = rule_tp.apply(_reasons, axis=1)
    rule_tp["expected_included"] = rule_tp.apply(
        lambda r: bool(EXPECTED_REASON[r["task_name"]]) and EXPECTED_REASON[r["task_name"]] in r["reasons"], axis=1
    )
    included = rule_tp[rule_tp["expected_included"]]
    fusion_alerts = preds[preds["pred_NeSy"] == 1]
    fusion_tp = fusion_alerts[fusion_alerts["y"] == 1]
    rule_fp = preds[(preds["pred_Rules"] == 1) & (preds["y"] == 0)].copy()
    rule_fp["reasons"] = rule_fp.apply(_reasons, axis=1)
    expl = {
        "rule_true_positives": int(len(rule_tp)),
        "expected_reason_included": int(len(included)),
        "expected_reason_included_with_additional_reasons": int((included["reasons"].str.len() > 1).sum()),
        "rule_true_positives_without_expected_reason": rule_tp.loc[~rule_tp["expected_included"], "reasons"].map(";".join).value_counts().to_dict(),
        "rule_false_alarms": int(len(rule_fp)),
        "rule_false_alarms_step_omission_only": int(rule_fp["reasons"].map(lambda r: all(x.startswith("Step_") for x in r)).sum()),
        "rule_false_alarms_postcondition_only": int(rule_fp["reasons"].map(lambda r: all(x.startswith("Post_") for x in r)).sum()),
        "fusion_alerts": int(len(fusion_alerts)),
        "fusion_alerts_with_rule_reason": int((fusion_alerts["pred_Rules"] == 1).sum()),
        "fusion_true_positives": int(len(fusion_tp)),
        "fusion_true_positives_with_rule_reason": int((fusion_tp["pred_Rules"] == 1).sum()),
    }
    (out / "explanations.json").write_text(json.dumps(expl, indent=2), encoding="utf-8")

    # 4. Run-to-run variation over the saved seeds
    thr = float(summary.get("decision_threshold", 0.5))
    seeds = summary["seeds"]
    var: Dict[str, Any] = {"seeds": seeds, "note": "Seeds share the outer folds and change initialisation and the inner validation split."}
    for name, col in [("GRU", "score_GRU_s{}"), ("GRU-motion", "score_GRU-motion_s{}")]:
        f1s, bas = [], []
        for s in seeds:
            p = (preds[col.format(s)] >= thr).astype(int).to_numpy()
            f1s.append(float(f1_score(y, p)))
            bas.append(float(balanced_accuracy_score(y, p)))
        var[name] = {"f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)), "f1": f1s,
                     "balanced_accuracy_min": float(min(bas)), "balanced_accuracy_max": float(max(bas)), "balanced_accuracy": bas}
    f1s = []
    for s in seeds:
        p = ((preds[f"score_GRU_s{s}"] >= thr) | (preds["pred_Rules"] == 1)).astype(int).to_numpy()
        f1s.append(float(f1_score(y, p)))
    var["NeSy"] = {"f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)), "f1": f1s}
    (out / "seed_variation.json").write_text(json.dumps(var, indent=2), encoding="utf-8")

    # 5. First recorded analog readings, aggregated per participant (descriptive)
    eps = build_episodes(cfg)
    ep = eps.episodes
    ev = eps.events.dropna(subset=["value_num"])
    first = ev.groupby(["episode_id", "sensor"])["value_num"].first().reset_index()
    first = first.merge(ep[["episode_id", "participant", "has_error", "task_name"]], on="episode_id")
    fr: Dict[str, Any] = {"note": "First recorded reading of a channel in an episode, median per participant; descriptive only."}
    for sensor, task in [("AD1-A", "Cook"), ("AD1-B", None), ("AD1-C", None)]:
        sub = first[first["sensor"] == sensor]
        if task:
            sub = sub[sub["task_name"] == task]
        agg = sub.groupby(["participant", "has_error"])["value_num"].median().reset_index()
        entry: Dict[str, Any] = {"scope": f"{task} episodes" if task else "all episodes in which the channel reported"}
        for flag, key in [(True, "error_condition"), (False, "correct_condition")]:
            v = agg.loc[agg["has_error"] == flag, "value_num"]
            entry[key] = {"participants": int(len(v)), "median": float(v.median()),
                          "q1": float(v.quantile(0.25)), "q3": float(v.quantile(0.75))}
        fr[sensor] = entry
    (out / "first_readings.json").write_text(json.dumps(fr, indent=2), encoding="utf-8")

    # 6. Post-hoc isolation-forest cutoff sensitivity (same folds and features)
    feats = episode_features(eps).loc[ep["episode_id"]]
    cols = [c for c in feats.columns if not c.startswith("task_")]
    fold_of = preds.set_index("episode_id")["fold"]
    y_of = preds.set_index("episode_id")["y"]
    quantiles = [0.0, 0.05, 0.10]
    iso_pred = {q: pd.Series(0, index=ep["episode_id"]) for q in quantiles}
    for fold in sorted(fold_of.unique()):
        te_ids = fold_of[fold_of == fold].index
        tr_ids = fold_of[fold_of != fold].index
        for task in TASK_ORDER:
            tr_ok = ep[ep["episode_id"].isin(tr_ids) & (ep["task_name"] == task) & ~ep["has_error"]]["episode_id"]
            te_t = ep[ep["episode_id"].isin(te_ids) & (ep["task_name"] == task)]["episode_id"]
            iso = IsolationForest(n_estimators=500, random_state=seed0).fit(feats.loc[tr_ok, cols].to_numpy())
            s_tr = iso.score_samples(feats.loc[tr_ok, cols].to_numpy())
            s_te = iso.score_samples(feats.loc[te_t, cols].to_numpy())
            for q in quantiles:
                iso_pred[q].loc[te_t] = (s_te < np.quantile(s_tr, q)).astype(int)
    yy = y_of.loc[ep["episode_id"]].to_numpy()
    iso_out = {
        "note": "Post-hoc sensitivity: cutoff = quantile q of the correct training executions' scores; q=0 is the reported baseline.",
        "matches_saved_q0": bool((iso_pred[0.0].to_numpy() == preds.set_index("episode_id").loc[ep["episode_id"], "pred_IForest"].to_numpy()).all()),
    }
    for q in quantiles:
        iso_out[f"{q:.2f}"] = _confusion(yy, iso_pred[q].to_numpy())
    (out / "iforest_cutoffs.json").write_text(json.dumps(iso_out, indent=2), encoding="utf-8")

    # 7. Strict-mode rule rerun (query failures raise) against saved predictions
    finals = final_analog_readings(eps)
    task_models = load_task_models()
    mismatches, missing_finals = 0, {}
    for fold in sorted(fold_of.unique()):
        te_ids = list(fold_of[fold_of == fold].index)
        tr = ep[ep["episode_id"].isin(fold_of[fold_of != fold].index)]
        th = calibrate_thresholds(task_models, tr, finals)
        det, _ = detect_errors(eps, te_ids, finals, th, task_models)
        saved = preds.set_index("episode_id").loc[det["episode_id"], "pred_Rules"].to_numpy()
        mismatches += int((det["rule_error"].astype(int).to_numpy() != saved).sum())
    for task, sensors in [("Cook", ["AD1-A"]), ("WashHands", ["AD1-B", "AD1-C"])]:
        ids = ep.loc[ep["task_name"] == task, "episode_id"]
        have = finals[finals["episode_id"].isin(ids) & finals["sensor"].isin(sensors)]["episode_id"].unique()
        missing_finals[task] = int(len(set(ids) - set(have)))
    # Inner validation splits of the main GRU runs (same procedure as the runner)
    from sklearn.model_selection import StratifiedGroupKFold

    ep_idx = ep.set_index("episode_id")
    val_err, val_ok = [], []
    for s in seeds:
        for fold in sorted(fold_of.unique()):
            tr = ep_idx.loc[list(fold_of[fold_of != fold].index)]
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(s))
            _, vi = next(inner.split(tr.index, tr["has_error"].astype(int), groups=tr["participant"]))
            v = tr.iloc[vi]
            val_err.append(int(v.loc[v["has_error"], "participant"].nunique()))
            val_ok.append(int(v.loc[~v["has_error"], "participant"].nunique()))
    rerun = {
        "prediction_mismatches": mismatches,
        "episodes_without_final_reading_on_monitored_sensor": missing_finals,
        "main_gru_validation_error_participants_min_max": [min(val_err), max(val_err)],
        "main_gru_validation_correct_participants_min_max": [min(val_ok), max(val_ok)],
        "main_gru_validation_splits_checked": len(val_err),
    }
    (out / "rule_rerun.json").write_text(json.dumps(rerun, indent=2), encoding="utf-8")

    # 8. Reproducibility manifest
    import rdflib
    import scipy
    import sklearn
    import torch

    raw_dir = Path(cfg["datasets"]["casas"]["raw_dir"])
    raw_files = sorted(raw_dir.glob("adl_*/*.csv"))
    agg = hashlib.sha256()
    for f in raw_files:
        agg.update(f.relative_to(raw_dir).as_posix().encode())
        agg.update(_sha256(f).encode())
    code_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=code_root, capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "unknown", True
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "libraries": {"torch": torch.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                      "numpy": np.__version__, "scipy": scipy.__version__, "rdflib": rdflib.__version__},
        "code_commit": commit,
        "code_working_tree_modified": dirty,
        "raw_data": {"files": len(raw_files), "events": int(len(eps.events)), "aggregate_sha256": agg.hexdigest()},
        "original_outputs_sha256": {f.name: _sha256(f) for f in sorted(res.glob("*")) if f.is_file()},
        "episodes": int(len(ep)), "participants": int(ep["participant"].nunique()),
        "max_events_per_episode": int(ep["n_events"].max()),
        "gru_max_seq_len": int(cfg["adl_errors"]["gru"]["max_seq_len"]),
        "fold_of_participant": preds.groupby("participant")["fold"].first().astype(int).to_dict(),
        "participants_in_more_than_one_fold": int((preds.groupby("participant")["fold"].nunique() > 1).sum()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Revision analyses written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

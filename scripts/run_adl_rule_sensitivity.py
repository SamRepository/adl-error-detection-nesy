"""
Rule sensitivity and ablation for ADL error detection
=====================================================
Re-runs only the symbolic branch under the participant-grouped folds of
run_adl_error_detection.py and reports:
  - off-threshold calibration: quantile q of correct training executions
    (q = 1.0 is the maximum used in the main results)
  - rule ablation: P1 (step omission) only, P2 (postcondition) only, both

Writes <output_dir>/rule_sensitivity.json.

Usage (Bash):
  python -m scripts.run_adl_rule_sensitivity --config config/adl_error_detection.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from adl_nesy.data.episodes import build_episodes, final_analog_readings
from adl_nesy.neural.utils import save_json
from adl_nesy.reasoning.procedural_rules import (
    calibrate_thresholds,
    detect_errors,
    load_task_models,
)
from adl_nesy.utils.config import load_config
from adl_nesy.utils.logging import setup_logging
from adl_nesy.utils.seed import set_global_seed

log = logging.getLogger(__name__)

QUANTILES = [0.90, 0.95, 1.0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sensitivity and ablation of the procedural rules.")
    ap.add_argument("--config", type=str, default="config/adl_error_detection.yaml")
    return ap.parse_args()


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "false_alarms": fp, "missed": fn}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging()
    acfg = cfg.get("adl_errors", {})
    seed0 = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed0)
    out_dir = Path(acfg.get("output_dir", "outputs/adl_errors"))

    eps = build_episodes(cfg)
    episodes = eps.episodes
    finals = final_analog_readings(eps)
    task_models = load_task_models()
    ids_all = episodes["episode_id"].tolist()
    y_all = episodes["has_error"].astype(int).to_numpy()
    groups = episodes["participant"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=int(acfg.get("k_folds", 5)), shuffle=True, random_state=seed0)
    folds = [([ids_all[i] for i in tr], [ids_all[i] for i in te]) for tr, te in splitter.split(ids_all, y_all, groups=groups)]

    y_of = dict(zip(ids_all, y_all))
    out: Dict[str, Any] = {"quantile": {}, "ablation": {}}
    for q in QUANTILES:
        rows: List[pd.DataFrame] = []
        for tr_ids, te_ids in folds:
            th = calibrate_thresholds(task_models, episodes[episodes["episode_id"].isin(tr_ids)], finals, quantile=q)
            res, _ = detect_errors(eps, te_ids, finals, th, task_models)
            rows.append(res)
        res = pd.concat(rows)
        y = res["episode_id"].map(y_of).to_numpy()
        p1 = (res["missing_steps"] != "").astype(int).to_numpy()
        p2 = (res["violated_postconditions"] != "").astype(int).to_numpy()
        out["quantile"][f"{q:.2f}"] = _metrics(y, ((p1 + p2) > 0).astype(int))
        if q == 1.0:
            out["ablation"] = {
                "P1": _metrics(y, p1),
                "P2": _metrics(y, p2),
                "P1+P2": _metrics(y, ((p1 + p2) > 0).astype(int)),
            }
        log.info("q=%.2f %s", q, out["quantile"][f"{q:.2f}"])
    log.info("Ablation %s", out["ablation"])
    save_json(out_dir / "rule_sensitivity.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

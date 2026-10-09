"""
Export paper assets for the revised manuscript
==============================================
Reads the original results of run_adl_error_detection.py and
run_adl_rule_sensitivity.py, plus the derived analyses of
derive_revision_analyses.py, and writes (no hand-typed numbers):

  macros.tex      one \\newcommand per reported number
  overall.tex     Table 2 (pooled results, grouped by information used)
  per_task_f1.tex Table 3 (F1 per task)

The figure and its tradeoff.json come from plot_tradeoff.py, which must run first.

The v2 assets (with the label-scarcity figure) are produced by
export_adl_paper_assets.py and are not used by the revision.

Usage:
  python -m scripts.export_revision_assets --results <outputs/adl_errors> \
      --derived <results_rev1> --dest <latex/generated>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from adl_nesy.utils.logging import setup_logging

log = logging.getLogger(__name__)

TASKS = [
    ("PhoneCall", "PhoneCall (wrong number)"),
    ("WashHands", "WashHands (water left on)"),
    ("Cook", "Cook (burner left on)"),
    ("Eat", "Eat (no medicine container)"),
    ("Clean", "Clean (no water)"),
]
DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
          "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export macros and tables for the revised paper.")
    ap.add_argument("--results", type=str, required=True, help="Original outputs/adl_errors")
    ap.add_argument("--derived", type=str, required=True, help="Output of derive_revision_analyses")
    ap.add_argument("--dest", type=str, required=True)
    return ap.parse_args()


def _name(*parts: str) -> str:
    out = []
    for p in parts:
        p = p.replace("-", " ").replace("_", " ").replace(".", " ")
        out.append("".join(w[:1].upper() + w[1:] for w in p.split()))
    return "".join(DIGITS.get(c, c) for c in "".join(out))


def _signed(v: float, nd: int = 3) -> str:
    # Math mode for a true minus sign; a rounded zero carries no sign
    return f"${0:.{nd}f}$" if abs(v) < 0.5 * 10 ** (-nd) else f"${v:+.{nd}f}$"


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}"


def write_macros(s: Dict[str, Any], per_task: pd.DataFrame, sens: Dict[str, Any], d: Dict[str, Any], dest: Path) -> None:
    m: Dict[str, str] = {}
    conf, fus, ex, sv, fr, iso, man = (d[k] for k in ["confusion", "fusion", "expl", "seed", "first", "iforest", "manifest"])

    m["NEpisodes"], m["NError"], m["NCorrect"] = str(s["n_episodes"]), str(s["n_error"]), str(s["n_correct"])
    m["NParticipants"], m["KFolds"], m["NSeeds"] = str(s["n_participants"]), str(s["k_folds"]), str(len(s["seeds"]))
    m["NEvents"] = f"{man['raw_data']['events']:,}"
    m["MaxEvents"] = str(man["max_events_per_episode"])
    m["MaxSeqLen"] = str(man["gru_max_seq_len"])

    for meth, c in conf.items():
        if meth == "trivial_all_error":
            continue
        for key in ["tp", "fp", "fn", "tn", "alerts"]:
            m[_name(meth, key)] = str(c[key])
        for key in ["precision", "recall", "f1"]:
            m[_name(meth, key)] = f"{c[key]:.3f}"
        m[_name(meth, "fpr")] = f"{100 * c['false_positive_rate']:.1f}"
    for meth, ci in s["f1_ci"].items():
        m[_name(meth, "f1 lo")], m[_name(meth, "f1 hi")] = f"{ci[0]:.2f}", f"{ci[1]:.2f}"
    for meth, p in s["pooled"].items():
        m[_name(meth, "balanced accuracy")] = f"{p['balanced_accuracy']:.3f}"
    m["TrivialFOne"] = f"{conf['trivial_all_error']['f1']:.3f}"
    for pair, v in s["paired_f1_diff_ci"].items():
        a, b = pair.split("-", 1)
        m[_name("diff", a, b)] = _signed(v["diff"])
        m[_name("diff", a, b, "lo")] = _signed(v["ci95"][0])
        m[_name("diff", a, b, "hi")] = _signed(v["ci95"][1])

    m["FusionAddedTP"] = str(fus["added_true_detections"])
    m["FusionAddedFP"] = str(fus["added_false_alarms"])
    m["FusionAddedPrecision"] = f"{fus['added_alert_precision']:.3f}"
    m["FusionAddedAlerts"] = str(fus["added_true_detections"] + fus["added_false_alarms"])

    m["RuleTP"] = str(ex["rule_true_positives"])
    m["ReasonIncluded"] = str(ex["expected_reason_included"])
    m["ReasonIncludedPct"] = _pct(ex["expected_reason_included"], ex["rule_true_positives"])
    m["ReasonIncludedExtra"] = str(ex["expected_reason_included_with_additional_reasons"])
    m["ReasonMissing"] = str(ex["rule_true_positives"] - ex["expected_reason_included"])
    m["RuleFaStep"] = str(ex["rule_false_alarms_step_omission_only"])
    m["RuleFaPost"] = str(ex["rule_false_alarms_postcondition_only"])
    m["FusionAlerts"] = str(ex["fusion_alerts"])
    m["FusionAlertsReason"] = str(ex["fusion_alerts_with_rule_reason"])
    m["FusionAlertsReasonPct"] = _pct(ex["fusion_alerts_with_rule_reason"], ex["fusion_alerts"])
    m["FusionTPReason"] = str(ex["fusion_true_positives_with_rule_reason"])
    m["FusionTPReasonPct"] = _pct(ex["fusion_true_positives_with_rule_reason"], ex["fusion_true_positives"])

    for meth in ["GRU", "NeSy", "GRU-motion"]:
        m[_name("seed", meth, "mean")] = f"{sv[meth]['f1_mean']:.3f}"
        m[_name("seed", meth, "sd")] = f"{sv[meth]['f1_sd']:.3f}"
    m["MotionBaMin"] = f"{sv['GRU-motion']['balanced_accuracy_min']:.3f}"
    m["MotionBaMax"] = f"{sv['GRU-motion']['balanced_accuracy_max']:.3f}"

    for sensor in ["AD1-A", "AD1-B", "AD1-C"]:
        for cond, tag in [("error_condition", "Err"), ("correct_condition", "Ok")]:
            e = fr[sensor][cond]
            m[_name("first", sensor, tag)] = f"{e['median']:.3f}"
            m[_name("first", sensor, tag, "n")] = str(e["participants"])

    for q, tag in [("0.00", "Min"), ("0.05", "Five"), ("0.10", "Ten")]:
        m[f"IsoF{tag}"] = f"{iso[q]['f1']:.3f}"
        m[f"IsoFa{tag}"] = str(iso[q]["fp"])
        m[f"IsoRec{tag}"] = f"{iso[q]['recall']:.2f}"

    for q, tag in [("0.90", "Ninety"), ("0.95", "NinetyFive"), ("1.00", "Max")]:
        m[f"SensF{tag}"] = f"{sens['quantile'][q]['f1']:.3f}"
        m[f"SensFa{tag}"] = str(sens["quantile"][q]["false_alarms"])
    for k, tag in [("P1", "POne"), ("P2", "PTwo"), ("P1+P2", "Both")]:
        m[f"Abl{tag}FOne"] = f"{sens['ablation'][k]['f1']:.3f}"
        m[f"Abl{tag}Recall"] = f"{sens['ablation'][k]['recall']:.2f}"

    th = pd.DataFrame(s["thresholds_per_fold"])
    for col in th.columns:
        m[_name("thr", col, "min")] = f"{th[col].min():.2f}"
        m[_name("thr", col, "max")] = f"{th[col].max():.2f}"
    m["LatencyRules"] = f"{s['latency_ms_per_episode']['Rules']:.1f}"
    m["LatencyGRU"] = f"{s['latency_ms_per_episode']['GRU']:.1f}"
    to = d["tradeoff"]
    for meth in ["RF", "LogReg", "GRU"]:
        m[_name("recall at rules fpr", meth)] = f"{to['curves'][meth]['recall_at_rules_fpr']:.2f}"
        m[_name(meth, "auroc")] = f"{to['curves'][meth]['auroc']:.3f}"
    for meth in ["Rules", "NeSy"]:
        p = to["points"][meth]
        m[_name(meth, "recall lo")], m[_name(meth, "recall hi")] = (f"{v:.2f}" for v in p["recall_ci95"])
        m[_name(meth, "fpr lo")], m[_name(meth, "fpr hi")] = (f"{100 * v:.1f}" for v in p["fpr_ci95"])
    for task, t in to["per_task"].items():
        for key in ["tp_rules", "tp_gru_added", "fp_rules", "fp_gru_added", "missed_by_fusion"]:
            m[_name("task", task, key)] = str(t[key])
    m["MissingFinalWash"] = str(d["rerun"]["episodes_without_final_reading_on_monitored_sensor"]["WashHands"])

    for _, r in per_task.iterrows():
        m[_name("task", r["task"], r["method"], "f1")] = f"{r['f1']:.2f}"

    lib = man["libraries"]
    m["VerPython"], m["VerTorch"] = man["python"], lib["torch"].split("+")[0]
    m["VerSklearn"], m["VerRdflib"] = lib["scikit-learn"], lib["rdflib"]

    lines = ["% Auto-generated by scripts/export_revision_assets.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (dest / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %d macros", len(m))


def write_tables(s: Dict[str, Any], conf: Dict[str, Any], per_task: pd.DataFrame, dest: Path) -> None:
    def row(name: str, meth: str, ci: Optional[list], auroc: Optional[float]) -> str:
        c = conf[meth]
        f1 = f"{c['f1']:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
        au = f"{auroc:.3f}" if auroc is not None else "--"
        return (f"{name} & {c['tp']} & {c['fp']} & {c['precision']:.3f} & {c['recall']:.3f} & {f1} & "
                f"{100 * c['false_positive_rate']:.1f} & {au} \\\\")

    pooled = s["pooled"]
    group = lambda text: r"\multicolumn{8}{@{}l}{\emph{" + text + r"}} \\"
    lines = [
        r"\begin{tabular}{@{}l@{\hspace{5pt}}rrccccc@{}}",
        r"\toprule",
        r"Method & TP & FP & Prec. & Rec. & F1 [95\% CI] & FPR (\%) & AUROC \\",
        r"\midrule",
        group("Rules (design informed by the full corpus; thresholds from correct executions)"),
        row("Procedural rules", "Rules", s["f1_ci"]["Rules"], None),
        group("Fitted on correct training executions only"),
        row("Isolation forest, per task", "IForest", s["f1_ci"]["IForest"], pooled["IForest"]["auroc"]),
        group("Supervised, fitted on error-labelled training folds"),
        row("Logistic regression", "LogReg", s["f1_ci"]["LogReg"], pooled["LogReg"]["auroc"]),
        row("Random forest", "RF", s["f1_ci"]["RF"], pooled["RF"]["auroc"]),
        row("GRU", "GRU", s["f1_ci"]["GRU"], pooled["GRU"]["auroc"]),
        group("Optional decision-level fusion"),
        row("Rules $\\lor$ GRU", "NeSy", s["f1_ci"]["NeSy"], None),
        r"\midrule",
        group("Controls"),
        row("GRU, motion events only", "GRU-motion", s["f1_ci"]["GRU-motion"], pooled["GRU-motion"]["auroc"]),
        row("Flag every episode", "trivial_all_error", None, None),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (dest / "overall.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = ["Rules", "IForest", "LogReg", "RF", "GRU", "NeSy"]
    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"Task (prescribed error) & Rules & IForest & LogReg & RF & GRU & Rules $\lor$ GRU \\",
        r"\midrule",
    ]
    for task, label in TASKS:
        sub = per_task[per_task["task"] == task].set_index("method")
        vals = [round(float(sub.at[c, "f1"]), 2) for c in cols]
        best = max(vals)
        cells = [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in vals]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (dest / "per_task_f1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    setup_logging()
    res, der, dest = Path(args.results), Path(args.derived), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    summary = load(res / "summary.json")
    sens = load(res / "rule_sensitivity.json")
    per_task = pd.read_csv(res / "per_task.csv")
    derived = {
        "confusion": load(der / "confusion.json"),
        "fusion": load(der / "fusion_tradeoff.json"),
        "expl": load(der / "explanations.json"),
        "seed": load(der / "seed_variation.json"),
        "first": load(der / "first_readings.json"),
        "iforest": load(der / "iforest_cutoffs.json"),
        "rerun": load(der / "rule_rerun.json"),
        "manifest": load(der / "manifest.json"),
        "tradeoff": load(der / "tradeoff.json"),
    }
    write_macros(summary, per_task, sens, derived, dest)
    write_tables(summary, derived["confusion"], per_task, dest)
    log.info("Revision assets written to %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Episode-level view of the CASAS Kyoto ADL error/no-error corpus.

Each raw file (one participant performing one scripted task) is one
*episode*. Episodes are the unit of analysis for task-execution error
detection: the label is whether the scripted execution contained the
task's injected error (Cook & Schmitter-Edgecombe, 2009).

Analog channels are discretised into level tokens so that the neural
model sees a closed vocabulary. Channel roles were established from the
data: AD1-A is the burner (idle ~2.80-2.85, rises to ~3.3 when heating);
AD1-B and AD1-C are the two sink water-flow channels (idle ~0.03).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import numpy as np
import pandas as pd

from adl_nesy.data.kyoto_loader import load_kyoto_adl_errors_events

log = logging.getLogger(__name__)

BURNER_SENSORS = ("AD1-A",)
WATER_SENSORS = ("AD1-B", "AD1-C")
ANALOG_SENSORS = BURNER_SENSORS + WATER_SENSORS


class EpisodeSet(NamedTuple):
    episodes: pd.DataFrame  # one row per episode
    events: pd.DataFrame    # raw events with an episode_id column


def build_episodes(cfg: Dict[str, Any]) -> EpisodeSet:
    """Group the Kyoto ADL event stream into one episode per source file."""
    casas = cfg.get("datasets", {}).get("casas", {})
    raw_dir = Path(casas.get("raw_dir", "data/raw/casas_kyoto_adl"))
    file_globs = casas.get("file_globs", ["adl_error/*.csv", "adl_noerror/*.csv"])

    # ISO8601 keeps whole-second rows (the phone sensor) that the loader's
    # default format inference drops.
    events = load_kyoto_adl_errors_events(raw_dir, file_globs=file_globs, timestamp_format="ISO8601")
    events["episode_id"] = events["dataset_split"] + "/" + events["source_file"].str.replace(".csv", "", regex=False)
    events["value_num"] = pd.to_numeric(events["value"], errors="coerce")
    events = events.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for ep_id, ev in events.groupby("episode_id", sort=True):
        first = ev.iloc[0]
        rows.append(
            {
                "episode_id": ep_id,
                "participant": first["participant"],
                "task_id": int(first["task_id"]),
                "task_name": first["task_name"],
                "has_error": bool(first["has_error"]),
                "start_time": ev["timestamp"].min(),
                "end_time": ev["timestamp"].max(),
                "duration_s": float((ev["timestamp"].max() - ev["timestamp"].min()).total_seconds()),
                "n_events": int(len(ev)),
            }
        )

    episodes = pd.DataFrame(rows).sort_values("episode_id").reset_index(drop=True)
    log.info(
        "Built %d episodes (%d with error) from %d participants",
        len(episodes),
        int(episodes["has_error"].sum()),
        episodes["participant"].nunique(),
    )
    return EpisodeSet(episodes=episodes, events=events)


def tokenize_event(sensor: str, value: str, value_num: float, level_cfg: Dict[str, Any]) -> str:
    """Map one raw event to a closed-vocabulary token."""
    if sensor in BURNER_SENSORS and not np.isnan(value_num):
        base = float(level_cfg.get("burner_idle", 2.80))
        step = float(level_cfg.get("burner_step", 0.1))
        level = int(np.clip(round((value_num - base) / step), 0, int(level_cfg.get("max_level", 6))))
        return f"{sensor}:L{level}"
    if sensor in WATER_SENSORS and not np.isnan(value_num):
        step = float(level_cfg.get("water_step", 0.1))
        level = int(np.clip(round(value_num / step), 0, int(level_cfg.get("max_level", 6))))
        return f"{sensor}:L{level}"
    return f"{sensor}:{value}"


def episode_token_sequences(
    eps: EpisodeSet,
    level_cfg: Dict[str, Any],
    *,
    include_task_token: bool = True,
    sensor_prefixes: tuple[str, ...] | None = None,
) -> Dict[str, tuple[List[str], List[float]]]:
    """Return {episode_id: (tokens, seconds_from_episode_start)}.

    ``sensor_prefixes`` restricts the sequence to sensors whose id starts with
    one of the prefixes (used for the motion-only confound control).
    """
    out: Dict[str, tuple[List[str], List[float]]] = {}
    task_of = dict(zip(eps.episodes["episode_id"], eps.episodes["task_name"]))
    for ep_id, ev in eps.events.groupby("episode_id", sort=False):
        t0 = ev["timestamp"].min()
        if sensor_prefixes is not None:
            ev = ev[ev["sensor"].str.startswith(sensor_prefixes)]
        toks = [tokenize_event(s, v, n, level_cfg) for s, v, n in zip(ev["sensor"], ev["value"], ev["value_num"])]
        deltas = (ev["timestamp"] - t0).dt.total_seconds().tolist()
        if include_task_token:
            toks = [f"TASK:{task_of[ep_id]}"] + toks
            deltas = [0.0] + deltas
        out[ep_id] = (toks, [float(d) for d in deltas])
    return out


def episode_features(eps: EpisodeSet) -> pd.DataFrame:
    """Bag-of-sensors summary features per episode (classical baseline input)."""
    ev = eps.events
    counts = ev.pivot_table(index="episode_id", columns="sensor", values="value", aggfunc="count", fill_value=0)
    counts.columns = [f"n_{c}" for c in counts.columns]

    analog = ev[ev["sensor"].isin(ANALOG_SENSORS)].dropna(subset=["value_num"])
    last = analog.groupby(["episode_id", "sensor"])["value_num"].last().unstack()
    last.columns = [f"last_{c}" for c in last.columns]
    peak = analog.groupby(["episode_id", "sensor"])["value_num"].max().unstack()
    peak.columns = [f"max_{c}" for c in peak.columns]

    meta = eps.episodes.set_index("episode_id")[["duration_s", "n_events", "task_name"]]
    feats = meta.join(counts).join(last).join(peak)
    feats = pd.get_dummies(feats, columns=["task_name"], prefix="task", dtype=float)
    return feats.fillna(0.0).astype(float)


def final_analog_readings(eps: EpisodeSet) -> pd.DataFrame:
    """Last reading of each analog channel per episode (long format)."""
    analog = eps.events[eps.events["sensor"].isin(ANALOG_SENSORS)].dropna(subset=["value_num"])
    last = analog.groupby(["episode_id", "sensor"])["value_num"].last().reset_index()
    return last.rename(columns={"value_num": "final_value"})

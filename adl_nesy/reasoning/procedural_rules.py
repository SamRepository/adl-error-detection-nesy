"""
Procedural rules for ADL task-execution error detection.

Two generic rules operate over task models declared in
``adl_procedural_extension.ttl``:

* Step omission: an episode of a task that requires a step, where none of
  the step's evidencing sensors reported, is an ``ExecutionError`` with
  ``hasMissingStep``.
* Postcondition violation: an episode whose final reading on a monitored
  sensor exceeds the postcondition's off-threshold is an
  ``ExecutionError`` with ``violatesPostcondition``.

The rules are task-agnostic; all task knowledge lives in the ontology.
Off-thresholds are the only data-derived quantities and are calibrated on
correct (error-free) training executions only, so the symbolic detector
never sees an error example.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from adl_nesy.data.episodes import EpisodeSet
from adl_nesy.reasoning.rule_engine import NSIOT, fire_rules

log = logging.getLogger(__name__)

SOSA = Namespace("http://www.w3.org/ns/sosa/")

DEFAULT_EXTENSION_PATH = Path(__file__).resolve().parents[1] / "ontology" / "adl_procedural_extension.ttl"

_SOSA_PREFIX = "PREFIX sosa: <http://www.w3.org/ns/sosa/>\n"

PROCEDURAL_RULES: List[Tuple[int, str, str]] = [
    # Rule P1 — Step omission. A required step with no evidencing observation.
    (101, "Required step without sensor evidence implies execution error", _SOSA_PREFIX + """
        CONSTRUCT { ?ep a nsiot:ExecutionError ; nsiot:hasMissingStep ?step . }
        WHERE {
            ?ep   a nsiot:TaskEpisode ;
                  nsiot:executesTask ?task .
            ?task nsiot:requiresStep ?step .
            FILTER NOT EXISTS {
                ?step nsiot:evidencedBy ?s .
                ?ep   nsiot:observedBy ?s .
            }
        }
    """),

    # Rule P2 — Postcondition violation. A device still above its
    # off-threshold when the episode ends.
    (102, "Device left on at task end implies execution error", _SOSA_PREFIX + """
        CONSTRUCT { ?ep a nsiot:ExecutionError ; nsiot:violatesPostcondition ?post . }
        WHERE {
            ?ep   a nsiot:TaskEpisode ;
                  nsiot:executesTask ?task .
            ?task nsiot:hasPostcondition ?post .
            ?post nsiot:monitoredBy ?s ;
                  nsiot:offThreshold ?th .
            ?obs  nsiot:isFinalObservationOf ?ep ;
                  sosa:madeBySensor ?s ;
                  sosa:hasSimpleResult ?v .
            FILTER (?v > ?th)
        }
    """),
]


def _episode_iri(episode_id: str) -> URIRef:
    return NSIOT["ep_" + re.sub(r"[^A-Za-z0-9]", "_", episode_id)]


def load_task_models(extension_path: Path = DEFAULT_EXTENSION_PATH) -> Graph:
    g = Graph()
    g.parse(str(extension_path), format="turtle")
    return g


def postcondition_monitors(task_models: Graph) -> Dict[str, Tuple[str, List[str]]]:
    """Return {postcondition_local_name: (task_name, [sensor ids])}."""
    q = """
        SELECT ?task ?post ?s WHERE {
            ?task nsiot:hasPostcondition ?post .
            ?post nsiot:monitoredBy ?s .
        }
    """
    out: Dict[str, Tuple[str, List[str]]] = {}
    for task, post, s in task_models.query(q, initNs={"nsiot": NSIOT}):
        post_name = str(post).split("#")[-1]
        task_name = str(task).split("#")[-1].replace("Task_", "")
        out.setdefault(post_name, (task_name, []))[1].append(str(s).split("#")[-1])
    return out


def calibrate_thresholds(
    task_models: Graph,
    train_episodes: pd.DataFrame,
    finals: pd.DataFrame,
    quantile: float = 1.0,
) -> Dict[str, float]:
    """Off-threshold per postcondition = highest final reading among correct
    training executions of the owning task (error episodes are excluded).
    ``quantile`` < 1 uses that quantile instead of the maximum (sensitivity
    analysis)."""
    monitors = postcondition_monitors(task_models)
    correct = train_episodes.loc[~train_episodes["has_error"], ["episode_id", "task_name"]]
    thresholds: Dict[str, float] = {}
    for post, (task_name, sensors) in monitors.items():
        ids = correct.loc[correct["task_name"] == task_name, "episode_id"]
        vals = finals.loc[finals["episode_id"].isin(ids) & finals["sensor"].isin(sensors), "final_value"]
        thresholds[post] = float(vals.quantile(quantile)) if len(vals) else float("inf")
    return thresholds


def build_episode_graph(
    eps: EpisodeSet,
    episode_ids: List[str],
    finals: pd.DataFrame,
    thresholds: Dict[str, float],
    task_models: Graph,
) -> Graph:
    """Populate a KG with task models, calibrated thresholds and episode facts."""
    g = Graph()
    for triple in task_models:
        g.add(triple)
    g.bind("nsiot", NSIOT)
    g.bind("sosa", SOSA)

    for post, th in thresholds.items():
        if th != float("inf"):
            g.add((NSIOT[post], NSIOT.offThreshold, Literal(th, datatype=XSD.double)))

    meta = eps.episodes.set_index("episode_id")
    wanted = set(episode_ids)
    sensors_per_ep = eps.events[eps.events["episode_id"].isin(wanted)].groupby("episode_id")["sensor"].unique()
    finals_sub = finals[finals["episode_id"].isin(wanted)]

    for ep_id in episode_ids:
        ep = _episode_iri(ep_id)
        row = meta.loc[ep_id]
        g.add((ep, RDF.type, NSIOT.TaskEpisode))
        g.add((ep, NSIOT.executesTask, NSIOT["Task_" + row["task_name"]]))
        g.add((ep, NSIOT.hasParticipant, NSIOT["person_" + row["participant"]]))
        for s in sensors_per_ep.get(ep_id, []):
            g.add((ep, NSIOT.observedBy, NSIOT[s]))

    for ep_id, sensor, value in finals_sub[["episode_id", "sensor", "final_value"]].itertuples(index=False):
        ep = _episode_iri(ep_id)
        obs = URIRef(str(ep) + "_final_" + re.sub(r"[^A-Za-z0-9]", "_", sensor))
        g.add((obs, RDF.type, SOSA.Observation))
        g.add((obs, SOSA.madeBySensor, NSIOT[sensor]))
        g.add((obs, SOSA.hasSimpleResult, Literal(float(value), datatype=XSD.double)))
        g.add((obs, NSIOT.isFinalObservationOf, ep))
    return g


def detect_errors(
    eps: EpisodeSet,
    episode_ids: List[str],
    finals: pd.DataFrame,
    thresholds: Dict[str, float],
    task_models: Graph,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the procedural rules; return per-episode verdicts and explanations."""
    t0 = time.perf_counter()
    g = build_episode_graph(eps, episode_ids, finals, thresholds, task_models)
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    summary = fire_rules(g, rules=PROCEDURAL_RULES)
    t_reason = time.perf_counter() - t1

    missing: Dict[str, List[str]] = {}
    for ep, step in g.subject_objects(NSIOT.hasMissingStep):
        missing.setdefault(str(ep), []).append(str(step).split("#")[-1])
    violated: Dict[str, List[str]] = {}
    for ep, post in g.subject_objects(NSIOT.violatesPostcondition):
        violated.setdefault(str(ep), []).append(str(post).split("#")[-1])

    rows = []
    for ep_id in episode_ids:
        key = str(_episode_iri(ep_id))
        m = sorted(missing.get(key, []))
        v = sorted(violated.get(key, []))
        rows.append(
            {
                "episode_id": ep_id,
                "rule_error": bool(m or v),
                "missing_steps": ";".join(m),
                "violated_postconditions": ";".join(v),
            }
        )
    timing = {
        "kg_build_s": t_build,
        "reasoning_s": t_reason,
        "n_episodes": len(episode_ids),
        "n_triples": len(g),
        "rule_firings": summary["rule_firings"],
    }
    return pd.DataFrame(rows), timing

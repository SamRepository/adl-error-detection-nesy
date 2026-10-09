"""
Fixed-point rule engine over an rdflib graph.

Each rule is a SPARQL ``CONSTRUCT`` query given as ``(rule_id, label, query)``.
Rules are fired repeatedly until no new triples are produced. The two
procedural rules do not depend on each other, so a single pass would give the
same result; the loop is an implementation choice that allows dependent rules.
The engine evaluates SPARQL over an rdflib graph only: it does not load
``owl:imports`` and does not perform OWL reasoning. A failing rule query
raises an error.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from rdflib import Graph, Namespace

log = logging.getLogger(__name__)

NSIOT = Namespace("http://example.org/neuro-symbolic-iot#")

PREFIXES = """
PREFIX nsiot: <http://example.org/neuro-symbolic-iot#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX time:  <http://www.w3.org/2006/time#>
"""


def fire_rules(
    g: Graph,
    *,
    rules: List[Tuple[int, str, str]],
    max_iters: int = 10,
) -> Dict[str, Any]:
    """Fire all rules in a fixed-point loop. Mutates ``g`` in place.

    Returns per-rule firing counts and per-iteration deltas.
    """
    rule_firings = {rid: 0 for rid, _, _ in rules}
    iters: List[Dict[int, int]] = []
    total_added = 0

    for _ in range(max_iters):
        new_this_iter = 0
        per_rule_this_iter: Dict[int, int] = {}
        for rid, _, query in rules:
            try:
                triples = list(g.query(PREFIXES + query))
            except Exception as exc:  # rdflib SPARQL parse / runtime error
                # Fail loudly: a silently skipped rule would look like an
                # episode without errors.
                raise RuntimeError(f"Rule {rid} query failed") from exc
            added_for_rule = 0
            for triple in triples:
                if len(triple) != 3:
                    continue
                if triple not in g:
                    g.add(triple)
                    added_for_rule += 1
                    new_this_iter += 1
            if added_for_rule:
                rule_firings[rid] += added_for_rule
                per_rule_this_iter[rid] = added_for_rule
        iters.append(per_rule_this_iter)
        total_added += new_this_iter
        if new_this_iter == 0:
            break

    return {
        "iterations": len(iters),
        "inferred_triples": total_added,
        "rule_firings": rule_firings,
        "per_iteration": iters,
    }

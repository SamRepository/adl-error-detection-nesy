# Detecting Task-Execution Errors in Smart Homes Without Error Examples

Code, ontology extension and experiment scripts for the paper
*Detecting Task-Execution Errors in Smart Homes Without Error Examples: A Neuro-Symbolic Approach*
(S. Sellami, R. Bouaita; submitted to ICMSCT 2026).

The detector decides whether a scripted activity of daily living (ADL) was carried out
correctly. It combines two parts:

- **Symbolic part.** Tasks are modelled in an ontology as required steps (each evidenced
  by sensors) and device postconditions. Two generic SPARQL rules flag *step omissions*
  and *postcondition violations*. They use **no error-labelled data**.
- **Neural part.** A GRU sequence model covers errors that leave no direct sensor trace.

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Data

The experiments use the CASAS scripted-ADL corpus with and without activity errors
(Cook, Zenodo, [10.5281/zenodo.15712834](https://doi.org/10.5281/zenodo.15712834)).
It is not redistributed here. Download it and unzip it so that the folder looks like this:

```
data/raw/casas_kyoto_adl/adl_error/p17.t1.csv ...
data/raw/casas_kyoto_adl/adl_noerror/p01.t1.csv ...
```

## Reproduce the results

Run from the repository root:

```bash
# All methods, 5 GRU seeds, label-scarcity sweep (about 1 hour on a laptop CPU)
python -m scripts.run_adl_error_detection --config config/adl_error_detection.yaml --scarcity

# Rule sensitivity (off-threshold quantile) and rule ablation (seconds)
python -m scripts.run_adl_rule_sensitivity --config config/adl_error_detection.yaml

# LaTeX number macros, tables and Fig. 2 for the paper
python -m scripts.export_adl_paper_assets --results outputs/adl_errors --dest paper_assets
```

Every run uses seed 42 (GRU seeds 42 to 46) and participant-grouped 5-fold
cross-validation. Results are fully deterministic on CPU.

| Output (`outputs/adl_errors/`) | Paper |
|---|---|
| `summary.json`, `predictions.csv` | Table 2, explanation and confound statistics |
| `per_task.csv` | Table 3 |
| `scarcity.csv` | Fig. 2 |
| `rule_sensitivity.json` | Robustness of the rules |
| `confound.json` | Confound control |

## Layout

| Path | Content |
|---|---|
| `adl_nesy/ontology/adl_procedural_extension.ttl` | Task models: steps, evidencing sensors, postconditions (SOSA/SAREF-aligned) |
| `adl_nesy/reasoning/procedural_rules.py` | The two rules (SPARQL `CONSTRUCT`), episode KG builder, off-level calibration |
| `adl_nesy/reasoning/rule_engine.py` | Fixed-point rule execution over rdflib |
| `adl_nesy/data/` | Corpus loader and episode builder |
| `adl_nesy/neural/` | GRU model and training loop |
| `scripts/` | Experiment, sensitivity and export scripts |

## Notes on the corpus

- **Channel roles.** The dataset documentation lists AD1-C as the burner. In the data,
  **AD1-A** is the burner (idle about 2.8, rising to about 3.3 while cooking).
  AD1-B and AD1-C are the two water-flow channels.
- **Timestamps.** Rows without fractional seconds (all telephone events) are dropped if
  a single timestamp format is inferred from the first row. The episode builder parses
  timestamps as ISO 8601 and keeps all 11,586 events.

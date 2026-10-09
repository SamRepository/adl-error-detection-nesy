# Procedural Knowledge for Detecting Task-Execution Errors in Smart Homes

Code, ontology extension and experiment scripts for the paper
*Procedural Knowledge for Detecting Task-Execution Errors in Smart Homes: An Exploratory
Neuro-Symbolic Study* (S. Sellami, R. Bouaita; submitted to ICMSCT 2026).

The detector decides whether a scripted activity of daily living (ADL) was carried out
as prescribed.

- **Symbolic detector.** Tasks are declared in an ontology as required steps, each
  evidenced by sensors, and as device postconditions. Two generic SPARQL rules flag
  *step omissions* and *postcondition violations*, and every alert names its reason.
  Off-thresholds are estimated from correct training executions only.
- **Optional supervised branch.** A GRU sequence model, trained with error labels, is
  combined with the rules by a logical OR. This raises recall at the cost of more
  false alarms.

## Provenance and scope of the results

The task-model steps are constrained by the published protocol. They were **selected,
and the sensor channels assigned their roles, after exploratory analysis of the full
corpus**, including summaries that compare erroneous and correct executions. Condition
labels are used only to select the correct executions for threshold estimation; inference
uses no labels. Because participants held out in each fold had already contributed to
the design, the reported results are **exploratory** and may be optimistic. They are not
an independent estimate of performance on unseen people.

## Setup

Python 3.10 or later (results were produced with Python 3.12.4, PyTorch 2.9.1 CPU,
scikit-learn 1.8.0, pandas 2.3.3, NumPy 2.3.5, SciPy 1.16.3, rdflib 7.6.0).

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
# 1. All methods, 5 GRU seeds (about 30 minutes on a laptop CPU)
python -m scripts.run_adl_error_detection --config config/adl_error_detection.yaml

# 2. Rule threshold sensitivity and rule ablation (seconds)
python -m scripts.run_adl_rule_sensitivity --config config/adl_error_detection.yaml

# 3. Derived analyses: confusion counts, fusion trade-off, explanation coverage,
#    seed variation, first recorded readings, isolation-forest cutoffs,
#    strict rule rerun, reproducibility manifest (no retraining)
python -m scripts.derive_revision_analyses --config config/adl_error_detection.yaml \
    --results outputs/adl_errors --out outputs/revision

# 4. Trade-off figure (Fig. 2) and its numbers, from the saved predictions
python -m scripts.plot_tradeoff --results outputs/adl_errors     --derived outputs/revision --dest paper_assets

# 5. LaTeX number macros and tables used by the paper (needs step 4)
python -m scripts.export_revision_assets --results outputs/adl_errors \
    --derived outputs/revision --dest paper_assets
```

Every run uses seed 42 (GRU seeds 42 to 46) and participant-grouped 5-fold
cross-validation. Results are deterministic on CPU.

| Output | Paper |
|---|---|
| `outputs/adl_errors/predictions.csv`, `summary.json` | Table 2, paired differences, CIs |
| `outputs/adl_errors/per_task.csv` | Table 3 |
| `outputs/adl_errors/rule_sensitivity.json` | Rule robustness |
| `outputs/revision/*.json` | Fusion trade-off, explanations, confound checks, manifest |
| `paper_assets/fig_tradeoff.pdf`, `outputs/revision/tradeoff.json` | Fig. 2 (operating points, detections per task) |

`--scarcity` (step 1) runs an additional label-scarcity sweep that is **not used in the
paper**. With 1–2 error-labelled participants, its inner validation split often contains
no error episodes, so best-epoch selection does not measure error detection (a warning is
logged). Because the rules were designed with full-corpus error data, the sweep also
cannot measure end-to-end label efficiency. `scripts/export_adl_paper_assets.py`
reproduces the assets of an earlier draft that included this sweep.

## Layout

| Path | Content |
|---|---|
| `adl_nesy/ontology/adl_procedural_extension.ttl` | Task models: steps, evidencing sensors, postconditions (SOSA/SAREF-aligned) |
| `adl_nesy/reasoning/procedural_rules.py` | The two rules (SPARQL `CONSTRUCT`), episode KG builder, threshold estimation |
| `adl_nesy/reasoning/rule_engine.py` | Fixed-point SPARQL execution over rdflib (no OWL reasoning; failing queries raise) |
| `adl_nesy/data/` | Corpus loader and episode builder |
| `adl_nesy/neural/` | GRU model and training loop |
| `scripts/` | Experiment, sensitivity, derivation and export scripts |

## Semantics and limits of the rules

- Task identity and episode boundaries come from the dataset.
- A step counts as evidenced if any of its sensors reported any event, including an
  analog change report. No event means missing evidence (closed world). This can confuse
  a sensor that did not record with an omitted action. A medicine-container event shows
  interaction with the container, not that medication was taken.
- A postcondition is checked only when the episode has a final reading on a monitored
  sensor (one WashHands episode has none). A postcondition without calibration data
  gets an infinite threshold and never fires.
- Labels are the participants' assigned experimental conditions, not adjudicated errors.

## Notes on the corpus

- **Channel roles.** The dataset documentation lists AD1-C as the burner. In the
  recordings, **AD1-A** behaves as the burner (about 2.8, rising to about 3.3 while
  cooking); AD1-B and AD1-C behave as the two water-flow channels. These roles were
  assigned after inspecting recordings from both conditions.
- **Timestamps.** Rows without fractional seconds (all telephone events) are dropped if
  a single timestamp format is inferred from the first row. The episode builder parses
  timestamps as ISO 8601 and keeps all 11,586 events.

## License

Apache License 2.0, see `LICENSE`. The CASAS dataset is subject to its own terms on Zenodo.

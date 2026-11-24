# Reproducing the railway report classification study

## Evidence and terminology

The experiment classifies an adapted three-category composite index from completed FRA-derived accident reports. It is a retrospective task. The archived numerical evidence is identified by commit `2e5f42c9429c92847fbe9b66b69cb61b562204bd`.

The final test set contains 8,935 incidents from the available 2021-2025 records. Training contains 99,784 incidents through 2017, and validation contains 6,188 from 2018-2020. Cross-period narrative components retain incidents in their latest represented partition. This does not remove every repetition within a retained partition.

`gated` in filenames denotes Vector Calibration Fusion (VCF). The three neural configurations use seeds 62, 63 and 64. The same test incidents are used by all thirteen fitted classifiers.

## 1. Verify the saved predictions on CPU

Use Python 3.10 for the small verification environment. The original training environment used Python 3.8.20.

```bash
python -m pip install -r requirements-verify.txt
python scripts/verify_results.py
```

Expected final line:

```text
PASS: all 13 final prediction files
```

This checks the metrics file's recorded hash, the exact expected classifier set, common incident identities and labels, 8,935 unique test incidents, finite normalized probabilities, predicted classes, accuracy, confusion matrices and macro-F1. It requires neither raw reports nor a GPU.

## 2. Recalculate the summary and conditional interval

Run from the repository root:

```bash
python analysis/v14/summarize_final.py --repo . --out work/check_summary
```

The outputs are `final_summary.csv` and `bootstrap.json`. Compare them with `analysis/v14/results/`. The mean VCF-minus-text difference is 0.0776527 percentage points. The paired component-bootstrap 95% interval is -0.343590 to 0.480287 percentage points, conditional on fixed fitted models. It does not measure uncertainty from fresh training or establish equivalence.

## 3. Recalculate the target and narrative audit

The raw CSV is stored inside `datanew_snapshot.zip`. The audit reads the ZIP directly:

```bash
python analysis/v14/audit_target.py --repo . --raw-zip datanew_snapshot.zip --out work/check_target
```

Use a new target-audit output directory. The script checks the archived raw snapshot, cohort and split records, rebuilds the labels and writes five files: `audit.json`, `mention_summary.csv`, `label_triggers.csv`, `descriptive_subgroups.csv` and `incident_screen.csv`.

The archived audit reproduces 114,916 stable candidate labels with zero mismatches, including nine incidents later excluded during cross-period filtering. Narrative keyword flags are descriptive, include negations, and do not prove that a report discloses its target. All audits and subgroups are post-test analyses.

## 4. Prepare a fresh training environment

Saved-prediction checks and audit recalculation do not require training. For training, use the recorded Python 3.8.20 environment and the core versions in `requirements.txt`. Install PyTorch 2.0.0+cu118 separately from its official wheel source. `environment/pip-freeze.txt` is a collected package inventory, not a complete environment lock.

Extract the raw snapshot into `data/raw/` from the repository root:

```bash
mkdir -p data/raw
unzip -n datanew_snapshot.zip -d data/raw
```

On Windows, use `Expand-Archive` instead of `unzip`:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Expand-Archive -LiteralPath datanew_snapshot.zip -DestinationPath data/raw
```

The extracted file must be `data/raw/datanew.csv`. Its SHA-256 is:

```text
9271f04027d907ae9c47d1de2d38a1b1ddf4eb64fcae177b9f62faa21be02e19
```

The snapshot contains 158,327,293 bytes. Its original export date and query are undocumented. A current live FRA export may differ and cannot establish reproduction of this snapshot.

Obtain the `FacebookAI/roberta-base` tokenizer, configuration and model files and place them in `pretrained_models/roberta-base/`. No pretrained weights are redistributed. Archived run protocols record the pretrained file hashes. The model modules are already included in `src/original_model/`; retain their attribution.

## 5. Train and evaluate a new reproduction

Use Bash for this sequence; each stage requires its expected output directories to be absent. Preserve earlier runs and select a new working copy rather than deleting experimental outputs.

```bash
python src/run_baselines.py --raw data/raw/datanew.csv --cohort data/incident_label_candidates.csv --duplicates data/duplicate_narrative_groups.jsonl --out work/baselines --train-cap 1000000 --val-cap 1000000
python src/finish_logistic.py --source work/baselines --out work/logistic_converged
for seed in 62 63 64; do
  python src/neural_run.py --source work/baselines --original src/original_model --plm pretrained_models/roberta-base --out work/seed_$seed --epochs 5 --batch-size 8 --seed "$seed"
done
python src/final_evaluate.py --validation-check
python src/final_evaluate.py
```

The portable evaluator reads `work/baselines`, `work/logistic_converged`, and `work/seed_62` through `work/seed_64`. It verifies a reproduction against its own validation records before future-period evaluation. A new stochastic training run need not equal the archived scores. The historical evaluator preserves the original lab paths and is retained as a provenance record.

The test period has already been examined. Additional model development informed by these results must not describe the same period as an untouched holdout. Independent checkpoint replay and fresh full-training reproduction have not been established by the saved-output verification.

## 6. Source lineage and diagrams

The repository includes preprocessing, label, identity and input-audit scripts under `src/railway_*`. Their command-line help documents required inputs. The supplied cohort and duplicate files are the inputs to the documented training sequence. `results/development_baselines/split_manifest.csv` records the archived partition assignments.

The cohort flowchart is supplied as [PDF](../figures/cohort_flow.pdf), [600-dpi PNG](../figures/cohort_flow.png), and [editable TikZ](../figures/source/Figure_1_Cohort.tex). Compile the TikZ file with pdfLaTeX. These figure files describe the cohort; they do not change its records or labels.

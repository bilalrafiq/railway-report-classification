# Railway report classification: temporal reproducibility materials

This repository supports retrospective classification of an adapted three-category accident index using narratives and structured fields. It does not establish official severity grades, prospective prediction or a significant overall advantage for gated fusion.

## Software author

**Bilal Rafique**, PhD Scholar, Computer Science, Southwest Jiaotong University. Contact: bilalrafique@my.swjtu.edu.cn. See [AUTHORS.md](AUTHORS.md) for affiliations and original-code attribution.

## Evidence
The final held-out cohort contains 8,935 cases dated 2021–2025 as available in the supplied extract. Three-seed test macro-F1: text-only 0.818549 ± 0.002492; concatenation 0.817544 ± 0.004510; gated 0.819326 ± 0.002918 (sample SD). All runs and predictions are included. The test period has now been evaluated; subsequent work must not describe it as an untouched holdout.

A repository is supporting reproducibility evidence, not proof of scientific validity or journal acceptance.

## Verify saved results without training
Install numpy and scikit-learn, then run:

```bash
python scripts/verify_results.py
```

## Reproduce training
The original environment used Python 3.8.20, torch 2.0.0+cu118, transformers 4.46.3 and sklearn 1.3.2 on an RTX 4090. See requirements.txt and the collected environment/pip-freeze.txt. Install PyTorch separately using its official matching wheel source. The repository does not install anything into a shared environment automatically. GPU runs are not guaranteed bitwise deterministic.

Obtain the exact author-supplied raw snapshot described in data/raw_snapshot.json. Extract datanew.csv into data/raw/. A current live FRA download may differ and must not silently replace the snapshot. Obtain FacebookAI/roberta-base model and tokenizer files in pretrained_models/roberta-base. The published run protocols list the exact pretrained file SHA256 values. No pretrained weights are redistributed here.

The original model.py and model_zoo.py must be present in src/original_model after server collection. Do not substitute another architecture. The included historical final evaluator documents the original lab paths; final_evaluate.py uses portable repository-relative paths.

```bash
python src/run_baselines.py --raw data/raw/datanew.csv --cohort data/incident_label_candidates.csv --duplicates data/duplicate_narrative_groups.jsonl --out work/baselines --train-cap 1000000 --val-cap 1000000
python src/finish_logistic.py --source work/baselines --out work/logistic_converged
for seed in 62 63 64; do
  python src/neural_run.py --source work/baselines --original src/original_model --plm pretrained_models/roberta-base --out work/seed_$seed --epochs 5 --batch-size 8 --seed "$seed"
done
python src/final_evaluate.py --validation-check
python src/final_evaluate.py
```

Each command requires a new output directory. The evaluator verifies internal consistency of a reproduction against its own validation records before testing. It does not require a new stochastic training run to equal the authors' scores. It preserves all three seeds and evaluates every comparator. Published exact predictions remain in results/.

## Data lineage
Included cohort candidates and duplicate groups permit rebuilding the exact input split from the raw snapshot. Source scripts for identity grouping, labels, input audit and preprocessing are included under src/railway_*. Their command-line help describes inputs. These audit stages document assumptions and exclusions and are distinct from the prediction-verification path. Results/development_baselines/split_manifest.csv records the published partition assignment.

## Data and rights
The official DOT Form 54 public-data portal is the upstream reference. The local snapshot was supplied by a coauthor; its original export query/date were not documented. Public availability of the source does not prove that every local field is unchanged. Verify provenance before releasing the raw snapshot. Keep the data snapshot separate from Git history and attach it to an immutable release or archive when its provenance is cleared.

No blanket license is assigned to coauthor code or third-party data in this preparation. The authors must select the code license and confirm rights before claiming unrestricted reuse. Retain applicable third-party notices. A public repository without an explicit code license is not a complete open-source release.

## Citation and release
The manuscript is not yet published; no paper DOI or acceptance is claimed. After publication, add the complete author-approved citation and paper DOI. Cite a tagged repository release and commit SHA; archive the release with a DOI if available. Do not insert a hypothetical URL into the manuscript. Record a release only after all files and the verification command are checked on GitHub.

## Packed data files
Large CSV/JSONL files are stored losslessly as .gz files. Run `python scripts/restore_data.py` before verification or training. The exact author-supplied raw snapshot is stored in private repository assets as numbered parts; this script verifies and reconstructs it. Keep the repository private until data redistribution and code licensing are settled.

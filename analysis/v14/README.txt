Post-test descriptive target audit, v14
Run from the repository root with Python 3.8+ (audit uses the standard library; summary requires numpy):
python analysis/v14/audit_target.py --repo . --raw-zip datanew_snapshot.zip --out work/target_audit_v14
python analysis/v14/summarize_final.py --repo . --out work/summary_v14
Use a new target-audit output directory. No training or prediction changes occur.
results/ contains the outputs generated on 16 November 2026. incident_screen.csv contains IDs and flags, not narrative text.
Keyword flags include negations and do not prove disclosure. All analyses are descriptive and post-test.
The bootstrap is conditional on fixed models and resamples linked components; its interval does not include training uncertainty.

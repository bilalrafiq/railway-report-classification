# Repository verification record

Checked 24 November 2026 at experiment commit `2e5f42c9429c92847fbe9b66b69cb61b562204bd`.

## Verified

- GitHub reports the repository as public.
- All 103 tracked file blobs matched GitHub's hashes in the checked snapshot.
- `AUTHORS.md` and `CITATION.cff` are present. The citation describes software separately from the manuscript byline.
- All thirteen final prediction files pass probability, identity, label and metric checks.
- Both summary files reproduce byte-for-byte: `final_summary.csv` and `bootstrap.json`.
- All five target-audit files reproduce byte-for-byte, including the incident-screen flags.
- The target audit reproduces 114,916 stable candidate labels with zero mismatches. The final training, validation and test counts are 99,784, 6,188 and 8,935.

| Neural configuration | Mean test macro-F1 | Sample SD |
|---|---:|---:|
| Text-only RoBERTa | 0.818549 | 0.002492 |
| Concatenation | 0.817544 | 0.004510 |
| VCF (`gated`) | 0.819326 | 0.002918 |

The mean VCF-minus-text difference is 0.0776527 percentage points. Its conditional paired component-bootstrap 95% interval is -0.343590 to 0.480287 points. No numerical corrections were required.

## Documentation and maintenance changes

The accompanying update clarifies raw-data setup, replaces stale private-repository instructions, links the post-test audit, supplies a CPU verification requirements file, and includes the cohort flowchart. It replaces the missing-model error message that referred to a nonexistent download script. The saved-result verifier now explicitly requires the full thirteen-classifier set.

Original predictions, metrics, protocols, raw data, split manifests, model equations and model settings are preserved. The script changes affect validation or diagnostic messages, rather than the training computation.

## Approved license update

No `LICENSE` file was present at the checked experiment commit. The authors have now approved the MIT License for the project code, and this update adds the root `LICENSE` file and records `license: MIT` in `CITATION.cff`. Existing author notices remain in place. The code license does not grant rights to raw data, pretrained models, or third-party material; applicable third-party terms continue to apply. No predictions, metrics, protocols, or model settings are changed by this license update.

## Remaining author decisions
- The raw snapshot ZIP is already tracked in the public repository. The authors should confirm its provenance, any local additions and redistribution permissions. Its export date and query remain undocumented.
- Preserve existing software attribution. Add a published-paper citation when bibliographic details become available; retain software attribution separately.
- A tagged release and archival DOI can be added after the release contents and any outstanding data or third-party permissions are settled. No release or DOI is invented here.

## Scope

This check verifies saved outputs and accessible repository records. It does not independently validate accident identity, regulatory equivalence of the adapted index, fresh model training, checkpoint replay or clinical/operational utility. The scratch runtime lacked PyTorch, so a new neural execution was not attempted. The code was inspected and parsed; no neural runtime failure is inferred from the absence of that dependency.

Repository URL: https://github.com/bilalrafiq/railway-report-classification

Official guidance on licensing: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

Official guidance on software citations: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files

# Reserved holdout: do not inspect for training or tuning

This directory holds fresh pilot holdout references and construction-time verification details. All reference QA happens before the timestamp in `../preparation.json`. After that boundary, ordinary tools may verify hashes but must not parse, display, export or evaluate these contents until after training and checkpoint selection.

The train/dev loader and exporter cannot open this split. This is an administrative safeguard, not encryption. Historical family exposure is disclosed in `../PILOT_DATA_SPEC.md`; the holdout is fresh at the statement level, not a previously unseen area of mathematics. No model has generated outputs on this split.

# Moved: Statistical Significance Guide

This content was merged into **Part V — Statistical significance testing** of:

- **[`thesis_overview.md`](thesis_overview.md)** — full-corpus / general methodology (seed-count power analysis, paired t-test / Wilcoxon / bootstrap CI formulas, Friedman multi-method test, reporting template, ad-hoc significance script).
- **[`thesis_overview_150_sentences.md`](thesis_overview_150_sentences.md)** — same content, with a status note confirming the 150-sentence pilot (DictaBERT + BEREL, all 6 fusion methods) completed with `--num-seeds 20`.

**Primary takeaway (corrected 2026-10-05):** paired significance testing (Wilcoxon signed-rank + t-test, Holm-adjusted) for every fusion method vs. Regular NER and vs. Cascade NER comes from the **`paired_fold_tests`** sheet of `unified_protocol_table.xlsx`, built by **`build_unified_protocol_table.py`**.

Earlier revisions of this file pointed at `journal_paired_fold_deltas` in `cross_comparison_*.xlsx`. **That sheet is not emitted.** `collect_oof_fold_long()` in `experiments/journal_results_export.py` skips every row whose `experiment_id` lacks `_oof`, so Exp01/Exp04 never get per-fold rows, `paired_fold_method_comparison()` finds neither baseline, and it returns an empty frame that the writer skips. The standalone builder recovers the baseline per-fold F1 from the OOF runs' own pooled predictions (no retraining) and runs the tests on identical tokens. Full explanation: Part V §V.1 of the two overviews below.

Edit **`thesis_overview.md`** or **`thesis_overview_150_sentences.md`** directly (Part V) — this file is kept only as a redirect.

"""Historical merge script (2026-09-30). Canonical docs: thesis_overview.md and thesis_overview_150_sentences.md.
Source files were replaced by redirect stubs; edit those two files directly."""
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "theisis overview.md"
FAIR = ROOT / "cross_comparison_fair_training_overview.md"
OOF = ROOT / "thesis_overview_fusion_oof_cv.md"
JOURNAL = ROOT / "thesis_overview_journal_paper_results.md"
P150 = ROOT / "thesis_overview_150_sentences_pilot.md"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()


def strip_leading_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip()
    return text


def main() -> None:
    main_body = strip_leading_h1(read(MAIN))
    fair_body = strip_leading_h1(read(FAIR))
    oof_body = strip_leading_h1(read(OOF))
    journal_body = strip_leading_h1(read(JOURNAL))
    p150_body = strip_leading_h1(read(P150))

    full_toc = """## Master table of contents

| Part | Section | Source (archived) |
|------|---------|-------------------|
| **I** | Pipeline methodology (~300 sentences default) | Former `theisis overview.md` |
| **II** | Fair training hyperparameters (Exp01 / Exp04) | Former `cross_comparison_fair_training_overview.md` |
| **III** | OOF router fusion (5-fold stratified CV) | Former `thesis_overview_fusion_oof_cv.md` |
| **IV** | Journal paper: one command, results & error analysis | Former `thesis_overview_journal_paper_results.md` |

**150-sentence pilot:** see companion file [`thesis_overview_150_sentences.md`](thesis_overview_150_sentences.md).

---
"""

    full_header = f"""# Thesis Overview (Full Labeled Corpus)

Consolidated thesis documentation for the **full NER dataset** (~300 sentences), cross-comparison runner, Exp01–Exp06/OOF fusion, fair training, and IEEE journal exports.

*Merged on {date.today().isoformat()} from all overview markdown files in this repository.*

{full_toc}
"""

    full_doc = "\n\n".join(
        [
            full_header,
            "# Part I — Pipeline methodology\n\n" + main_body,
            "# Part II — Fair training hyperparameters\n\n" + fair_body,
            "# Part III — OOF router fusion\n\n" + oof_body,
            "# Part IV — Journal paper workflow & results\n\n" + journal_body,
        ]
    )

    (ROOT / "thesis_overview.md").write_text(full_doc + "\n", encoding="utf-8")

    p150_toc = """## Master table of contents

| Part | Section |
|------|---------|
| **I** | 150-sentence pilot — CLI, data, experiments, metrics |
| **II** | Fair training (DictaBERT / BEREL pilot profile) |
| **III** | OOF fusion on 150 sentences (5 folds × ~30 test sentences) |
| **IV** | Journal one-command run (`--journal-paper`) for 150-sentence subset |

**Full-corpus math and global architecture:** [`thesis_overview.md`](thesis_overview.md) Part I.

---
"""

    p150_header = f"""# Thesis Overview — 150-Sentence Training Pilot

Consolidated documentation for the **150-sentence subset** (fixed `--subset-seed 42`), DictaBERT / BEREL cross-comparison, OOF routers, loss-weight calibration, and journal exports.

*Merged on {date.today().isoformat()} from `thesis_overview_150_sentences_pilot.md` and shared overview supplements.*

{p150_toc}
"""

    p150_doc = "\n\n".join(
        [
            p150_header,
            "# Part I — 150-sentence pilot\n\n" + p150_body,
            "# Part II — Fair training hyperparameters\n\n" + fair_body,
            "# Part III — OOF router fusion (150 sentences)\n\n" + oof_body,
            "# Part IV — Journal paper workflow (150-sentence example)\n\n" + journal_body,
        ]
    )

    (ROOT / "thesis_overview_150_sentences.md").write_text(p150_doc + "\n", encoding="utf-8")

    print("thesis_overview.md", len(full_doc.splitlines()), "lines")
    print("thesis_overview_150_sentences.md", len(p150_doc.splitlines()), "lines")


if __name__ == "__main__":
    main()

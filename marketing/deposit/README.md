# Zenodo deposit files

Everything in this folder is what gets attached to the Zenodo record. Upload all of it, or
just the first item if you want a minimal deposit — Zenodo requires at least one file and a
PDF is the convention for a Technical note.

## What to upload

| Order | File | Required? | Why |
|---|---|---|---|
| 1 | `Fifteen-Minutes-Ahead.pdf` | **Yes** | The report itself. This is what people cite, download, and read. 6 pages, A4 |
| 2 | `evaluation-results.csv` | Recommended | Every number in the report as tidy data, with denominators and event definitions. This is what makes the deposit reproducible rather than just readable |
| 3 | `evaluate_*.py`, `train_experiment.py` | Recommended | The scripts that produced those numbers |
| 4 | `Fifteen-Minutes-Ahead.html` | Optional | Source of the PDF. Include only if you want the deposit self-contained |

Zenodo marks the first uploaded file as the preview, so **upload the PDF first**.

## What NOT to upload here

- **The CGM data.** It is already published in the GitHub repository, and the separate
  Software DOI (see `../ZENODO-DEPOSIT.md` §7) archives that repository including the data.
  Duplicating 2 MB of readings into the publication record adds nothing and muddies what
  the record is for.
- **Model artefacts (`.onnx`).** Same reason — they belong to the software record.

Two DOIs, cleanly separated: one for the report, one for the code and data it describes,
cross-linked with `is supplemented by`.

## Regenerating the PDF

The PDF is derived from the published HTML report so the content stays single-sourced.
If the report changes, rebuild rather than hand-editing:

```bash
python build_print.py    # strips dark theme, animations, and web nav; adds print CSS + abstract
chrome --headless --disable-gpu --virtual-time-budget=20000 --no-pdf-header-footer \
  --print-to-pdf=Fifteen-Minutes-Ahead.pdf fifteen-minutes-ahead-print.html
```

`build_print.py` lives in the working scratchpad, not in this repository — it is a build
tool, not a deliverable. The print HTML it produces is included above so the PDF is
reproducible without it.

## Before you deposit

The report currently carries a DOI placeholder in its citation block. Sequence:

1. Deposit with the PDF as-is and reserve a DOI (Zenodo offers "Reserve DOI" before publishing)
2. Put the reserved DOI into the report's citation block
3. Rebuild the PDF
4. Replace the file in the draft
5. Publish

That gets you a report that cites its own DOI, which is the detail that makes a deposit
look deliberate rather than improvised.

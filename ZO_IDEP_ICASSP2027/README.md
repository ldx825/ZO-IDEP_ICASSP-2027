# ZO-IDEP ICASSP 2027 Paper

## Files
- `main.tex`: ICASSP-style paper with theory, algorithm, and measured results.
- `refs.bib`: verified bibliography.
- `CITATION_AUDIT.md`: source-by-source citation verification notes.
- `THEORY_WORKING_NOTE.md`: longer derivation notes used to build the paper.
- `REVIEW_AND_NOVELTY.md`: proof, novelty, collision, and claim-scope audit.
- `EXPERIMENT_REPORT.md`: protocols, aggregate results, and limitations (documents the deprecated prototype pipeline).
- `zo_idep/`: canonical experiment package (`src/` + `experiments/exp00--exp07`) used for the paper's tables and figures.
- `scripts/`: deprecated prototype pipeline; see `scripts/DEPRECATED.md`.
- `official_template/`: untouched ICASSP 2027 Paper Kit baseline.
- `spconf.sty`, `IEEEbib.bst`: working copies that hash-match the official files.

## Compile
```bash
make
```

Continuous build: `make watch`. Submission preflight: `make check`.

To reproduce the reported results (canonical pipeline):

```bash
python3 zo_idep/experiments/exp01_sparse_regression.py --trials 30 --iterations 300
python3 zo_idep/scripts/validate_results.py
```

The legacy `scripts/` commands reproduce an early prototype only; do not use them for the paper numbers.

The VS Code workspace uses LaTeX Workshop, autosave after one second,
build-on-save, the internal PDF viewer, and SyncTeX. Open `main.tex`, then run
`LaTeX Workshop: View LaTeX PDF`; double-click the PDF for reverse SyncTeX.

## Before submission
1. Replace author, affiliation, and funding placeholders.
2. Decide whether to include a nonquadratic/noisy benchmark in the submission.
3. Have a coauthor independently verify the theorem assumptions and proof chain.
4. Re-run literature search immediately before making any priority claim.
5. Check the current IEEE/ICASSP AI-content disclosure requirement.

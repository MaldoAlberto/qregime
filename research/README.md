# Research archive

This folder is the raw research log behind the paper and the `qregime`
package: every experiment script, dataset and intermediate result, exactly
as it was run. It is **not** part of the installable package and its code is
not maintained as a library -- it is here so that every number in the paper
can be traced back to the code that produced it.

Most comments and READMEs in here are in Spanish (the language the research
was done in). The main log is `README_research_log_es.md`; the decision-layer
log with the full history of findings, corrections and retractions is
`decision-layer/README.md`.

| Folder | Contents |
|---|---|
| `techniques/` | one folder per circuit-reduction technique, with its own scope and limits |
| `qec/` | from-scratch rotated surface-code simulation (Stim + PyMatching) and threshold fit |
| `decision-layer/` | the research versions of the advisor, estimators, stress tests and ML experiments |
| `benchmarks/` | the 98-instance, 9-family circuit dataset with measured ZX-calculus gain |
| `educational/` | an earlier explanatory write-up and a pipeline diagram |

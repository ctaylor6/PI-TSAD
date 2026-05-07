# Repository Strategy

This repository separates the quick, reproducible APS validation path from the
long-running full-part analysis path.

- `data/aps/` includes only the small CSVs required to reproduce the 4-fold
  APS example.
- `src/pi_tsad/` contains the maintainable package code.
- `examples/` contains short scripts users can run or adapt.
- `tests/` covers the core pieces that should stay stable as the method evolves.
- `LPBF_Random_Forrest/` is ignored as local source material and should not be
  pushed to GitHub.

For full-scale builds, train or load a model, then run `detect-part` against a
user-provided CSV. The command supports `--layer` and `--max-rows` so users can
test a small slice before starting a long run.

# Restructuring Report

## 1. Initial repository assessment

The starting folder was not a Git repository. Therefore:

- no branch, HEAD, remote, tracked-file list, or history existed;
- `git status --short`, `git branch --show-current`, and `git rev-parse HEAD`
  returned “not a git repository”;
- no baseline HEAD could be recorded; and
- `git init` was deliberately not run.

The implementation already had a coherent, importable root-package layout:

```text
backtest/ core/ data/ features/ metrics/ reports/ strategies/
```

`main.py` was the full research CLI, configs were under `configs/`, and tests
used Python `unittest`. The dependency manifest was a single
`requirements.txt`. Local market-data caches totaled about 53 MiB, while local
generated reports totaled about 6.3 GiB.

The original README contained substantial methodology and result history but
led with implementation chronology and strong numeric claims. It did not
clearly separate the historical 0.5-null FDR result from the final
majority-drift-aware gate.

## 2. Structural changes made

- Repositioned the project as a quantitative signal-falsification and
  reproducibility system.
- Preserved the stable root packages instead of forcing a `src/` migration.
- Added PEP 621 packaging and dependency groups in `pyproject.toml`.
- Added a network-free canonical synthetic entry point.
- Added a deterministic fictional OHLCV sample and generator.
- Added a machine-readable public result-audit generator.
- Added public methodology, validation, results, limitations, AI disclosure,
  data policy, security audit, and release checklist documentation.
- Added a read-only security-check script.
- Added minimal-permission GitHub Actions for lint, unit tests, and the
  synthetic smoke run.
- Expanded `.gitignore` to exclude raw/cache data, generated outputs, secrets,
  serialized artifacts, and local internal records.
- Applied Ruff formatting to 30 Python files and fixed four Ruff lint findings.

## 3. Files moved

None.

No source package, raw data, output artifact, notebook, or legacy work record
was moved. This avoided import churn and preserved local research paths.

## 4. Files added

### Root release metadata

- `.env.example`
- `pyproject.toml`
- `LICENSE`
- `CITATION.cff`
- `SECURITY_AUDIT.md`
- `RESTRUCTURING_REPORT.md`

### Public documentation

- `docs/methodology.md`
- `docs/validation.md`
- `docs/results.md`
- `docs/limitations.md`
- `docs/ai_assistance.md`
- `docs/public_release_checklist.md`
- `data/README.md`
- `data/sample/README.md`
- `notebooks/README.md`
- `reports/README.md`
- `reports/public/README.md`

### Reproducibility and release scripts

- `configs/example_config.yaml`
- `data/sample/synthetic_ohlcv.csv`
- `scripts/__init__.py`
- `scripts/generate_synthetic_sample.py`
- `scripts/run_experiment.py`
- `scripts/build_public_summary.py`
- `scripts/run_tests.sh`
- `scripts/security_check.sh`

### Public evidence tables

- `reports/public/verified_flagship_runs.csv`
- `reports/public/intraday_cost_survival.csv`

### CI and tests

- `.github/workflows/ci.yml`
- `tests/test_public_interface.py`

## 5. Files intentionally left unchanged

No methodology or historical output was changed to make a test pass.

The following remain local and unchanged in substance:

- all 95 Parquet cache files under `data/cache*`;
- all 511 generated files under `reports/output*`;
- all historical experiment configs except the addition of
  `configs/example_config.yaml`;
- legacy research summaries, progress logs, prompts, and generated HTML;
- the PIT membership data and its historical notes; and
- model, feature, label, splitter, and statistical formulas.

Ruff applied whitespace and line-wrapping changes across existing Python files.
Two non-formatting lint fixes moved a module docstring to its conventional
position in `reports/pit_comparison.py` and removed one unused test import.

## 6. Import or API compatibility changes

No package was renamed and no public callable signature was intentionally
changed.

`main.py` remains the full-data CLI and is also exposed as the installed
`equity-direction-research` console command. Its argument parser description
was changed from a mega-cap predictor description to an equity-signal
falsification description.

The new `scripts/run_experiment.py` is the canonical public smoke interface. It
uses the same feature, split, strategy, backtest, metric, and report modules on
fictional data.

## 7. Test results before changes

An initial `pytest -q` attempt failed because `pytest` was not installed. The
project's documented framework was actually `unittest`.

Canonical baseline:

```text
.venv/bin/python -m unittest discover -s tests -v
Ran 80 tests in 3.787s
OK
```

PyArrow emitted sandbox-specific CPU cache-size warnings during one Parquet
test, but the test passed.

## 8. Test results after changes

Final local result:

```text
.venv/bin/python -m unittest discover -s tests -v
Ran 86 tests
OK
```

Six tests were added for:

- active-row transaction-cost sign and basis-point conversion;
- test rows occurring after their fold's training end;
- fixed-seed random-baseline reproduction;
- next-bar execution-price alignment; and
- deterministic, explicitly marked public synthetic data; and
- Polygon HTTP-error traceback credential redaction.

## 9. Lint or static-analysis results

```text
ruff check .
All checks passed!

ruff format --check .
70 files already formatted

python -m compileall -q ...
passed

import check for all primary packages
passed
```

`mypy` was not configured or installed and was not introduced. `actionlint`
was not installed.

## 10. Reproduction commands verified

Verified locally:

```bash
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_experiment.py --config configs/example_config.yaml
.venv/bin/python scripts/build_public_summary.py
./scripts/security_check.sh
```

The synthetic smoke run produced:

```text
6 OOS summary rows
0 beats-majority-after-FDR passes
10 generated report files including data_summary.csv
```

Generated smoke outputs are under ignored `artifacts/smoke/`.

The full historical experiment was not rerun because it requires local vendor
data, provider assumptions, and substantial compute.

## 11. Documentation added

The public documentation now separately covers:

- research identity and negative-result interpretation;
- labels, features, strategies, execution timing, and data cleaning;
- walk-forward, purge, embargo, CPCV, overlap, and ESS behavior;
- corrected and drift-aware statistical results;
- cost sensitivity;
- data schema, adjustment, timezone, provider, and licensing policy;
- limitations and non-generalizability;
- AI-assisted development;
- security findings; and
- a staged public-release checklist.

## 12. Remaining owner decisions

1. **The repository owner must confirm licensing.** The textual scan found no
   third-party code notice, but ownership cannot be proven automatically.
2. **Data-provider rights require manual review.** Raw/cache data must remain
   excluded.
3. **The v1–v4 “442 combinations” denominator lacks one reconstructable
   manifest.** It is disclosed as a historical reported count.
4. **Citation identity is generic.** The owner should decide whether to add a
   verified public author name.
5. **Generated HTML and legacy internal records are excluded.** Publishing any
   of them requires a separate content and privacy review.

## 13. Recommended manual review

- Read `README.md`, `docs/results.md`, and every row of `reports/public/*.csv`.
- Confirm the five source summary checksums in the original local workspace.
- Confirm the six-timeframe implementation / five-timeframe archived-result
  distinction.
- Confirm that “0 final passes” uses `beats_majority_after_fdr`, not the legacy
  0.5-null field.
- Decide whether the manually assembled PIT membership notes are appropriate
  for public release.
- Confirm code ownership and the MIT license.
- Confirm citation author metadata.
- For every future change, stage only reviewed files and rerun the security
  script.
- Inspect `git status --short`, staged paths, staged sizes, and the complete
  staged diff before approving another commit.

## 14. Git release handoff

After the repository owner explicitly approved a local commit, the workspace
was initialized as a new `main` repository. Public files were staged by an
explicit allowlist, and the staged snapshot was checked for ignored private
records, credentials, personal paths, unsafe serialized files, and files larger
than 10 MiB before the first commit.

After a second explicit owner approval, a credential-free HTTPS `origin` was
configured and `main` was pushed to the new public GitHub repository. The
GitHub Actions matrix passed on Python 3.11 and 3.12, including installation,
lint, unit tests, and the synthetic smoke test.

No tag, release, pull request, or history rewrite beyond the pre-publication
noreply-author amend was performed. Future pushes should retain the same
security checks:

```bash
git remote -v
./scripts/security_check.sh
git push -u origin main
```

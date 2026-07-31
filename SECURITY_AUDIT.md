# Security Audit

## 1. Audit scope

Audit date: 2026-07-31

Scope: the local workspace prepared as a public portfolio candidate.

The audit included:

- redacted high-signal secret and credential-pattern scans;
- public-scope email and absolute-path scans;
- `.env`, private-key, notebook, serialized-object, and checkpoint discovery;
- raw/cache data and generated-output inventory;
- files larger than 10 MiB;
- dependency and GitHub Actions review; and
- attempted Git repository, remote, tracked-file, and history inspection.

The workspace had no pre-existing Git repository or history. It was initialized
on `main` only after the working-tree audit, and the exact first-commit snapshot
was then inspected separately. No remote is configured. The public content scan
excluded `.venv`, local data caches, ignored `reports/output*` trees, generated
`artifacts`, and legacy internal work records. Those excluded paths were
inventoried separately.

No scanner can establish that a repository is completely safe. This report is
limited to the named checks and current filesystem state.

## 2. Secret scan

### High-signal patterns

No matches were found in the public working-tree scope for:

- GitHub personal-access-token prefixes;
- OpenAI-style secret-key prefixes;
- AWS access-key prefixes;
- Slack token prefixes;
- PEM private-key headers.

The Polygon URL's runtime-variable interpolation and the deliberate fake key in
the security regression test are reported as credential-like query-parameter
candidates. Manual review confirmed that neither is a real embedded key.

Values were never printed by the scan.

### Generic credential candidates

Candidate lines were reported in `data/loader.py` and the Polygon security
regression test. Manual review found that the production candidates:

- retrieve `POLYGON_API_KEY` from the environment;
- interpolate the runtime variable into the provider URL; and
- carry that runtime variable across provider pagination.

No real credential is present. Alpaca credentials are likewise read from
`ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables. `.env.example`
contains blank placeholders only. The test candidates use an explicitly fake
fixture value to verify traceback redaction.

Polygon's endpoint requires the API key in the request query string. Provider
request and response-decoding failures are caught and replaced with a
URL-free `DataProviderError`; exception chaining is suppressed so the original
`HTTPError.url` is not rendered in a public traceback. A regression test
injects a key-bearing `HTTPError` and verifies that neither the exception
message nor its rendered traceback contains the key.

Severity: **Informational**

### Tooling limitation

`gitleaks` was not installed. The audit used the repository's redacting
`scripts/security_check.sh` and manual pattern scans instead. The script uses
`rg` when available and falls back to `grep -E`; both execution paths were
tested. If neither scanner is available, the script records a blocker rather
than reporting an apparently clean result.

## 3. Git history scan

Pre-initialization inspection established that this folder had no Git
repository or history. The new repository was created on `main`, the complete
first-commit file set was staged explicitly, and staged paths, contents, and
sizes were inspected before commit. No remote URL is configured.

At the point this report enters history, that history consists only of the
audited initial snapshot. If these files are later copied into an existing
repository, that repository's complete history must be scanned separately.

Severity: **Informational**

## 4. Personal-information scan

No email address was found in the public working-tree scope. `CITATION.cff`
uses the collective label “Project contributors” and contains no personal
email.

An ignored local implementation log contains a user-specific absolute
workspace path and a runtime installation path. Values are intentionally not
reproduced here. The file is excluded from the proposed public release but
remains on disk.

No telephone number, street address, student identifier, private server
hostname, or internal-IP candidate was identified by the performed public-scope
scan.

Severity: **Low**

## 5. Local absolute-path scan

No hard-coded macOS home, Linux home, or Windows user-profile path was found in
the proposed public scope after excluding the security script's own detection
pattern.

Runtime scripts derive repository paths from `__file__`; they do not encode the
current user's home directory. Runtime console output can naturally print the
resolved local path when commands are executed.

The ignored local implementation log contains two local-path occurrences and
should remain excluded.

Severity: **Low**

## 6. Raw-data and licensing scan

Local data inventory:

- 95 Parquet files under `data/cache*`;
- approximately 53 MiB across the data tree;
- price histories, Alpaca IEX intraday bars, FRED series, CBOE SKEW, and FINRA
  short-interest caches; and
- a manually researched PIT membership file under `configs/`.

All `data/cache/` and `data/cache_*` directories are ignored. No file was
deleted or moved.

The code license does not grant redistribution rights for provider data.
Provider terms and the provenance of the manually assembled universe table
require owner review before publication.

No copyright header, copied-source notice, third-party license file, or SPDX
marker was found in the project source outside `.venv`. That textual scan does
not prove original authorship. The added MIT license assumes that the repository
owner has rights to license the code; the owner must confirm this assumption.

Severity: **Medium**

## 7. Large-file scan

The workspace contains 31 files larger than 10 MiB under `reports/` and
`data/`. The report tree totals approximately 6.3 GiB and contains 511 local
generated-output files.

Largest examples:

| File | Approximate size |
|---|---:|
| `reports/output/predictions.csv` | 1.83 GB |
| `reports/output_pit_union_static/predictions.csv` | 641 MB |
| `reports/output_pit_current10/predictions.csv` | 334 MB |
| `reports/output_ext_macro_on_ab_t10y2y/predictions.csv` | 203 MB |
| `reports/output_ext_macro_on_ab_dgs10/predictions.csv` | 203 MB |

Every `reports/output*` path is ignored. `.gitignore` alone will not remove a
file that is force-added or already tracked in a future repository, so the
exact staged set must be inspected.

Severity: **Medium**

## 8. Serialized-artifact scan

No pickle, joblib, HDF5, PyTorch checkpoint, ONNX model, or similarly risky
serialized artifact was found outside `.venv`.

Parquet files were identified as data artifacts, not loaded as arbitrary Python
objects during the security scan.

Severity: **Informational**

## 9. Notebook-output scan

No `.ipynb` files were found. `notebooks/README.md` documents that future
notebooks must have outputs and sensitive data reviewed before release.

Several generated HTML reports remain locally. They are ignored because they
contain historical presentation material that was not re-audited line by line
for the public portfolio.

Severity: **Low**

## 10. Dependency and CI review

`pyproject.toml` separates core, provider, ML, boosting, plotting, development,
and notebook dependencies. Versions use bounded ranges rather than exact pins.
There is no lockfile, so a future installation is not bit-for-bit reproducible.

`.github/workflows/ci.yml`:

- declares `permissions: contents: read`;
- uses no repository secret;
- runs on Python 3.11 and 3.12;
- installs the public package and Ruff;
- runs lint, the network-free unit suite, and the synthetic smoke test; and
- does not run the historical experiment.

The YAML parsed locally. `actionlint` was unavailable, and the workflow has not
run remotely.

Severity: **Low**

## 11. Findings by severity

### Critical

None found within the performed scope.

### High

None found within the performed scope.

### Medium

1. No Git repository/history/tracked-file set exists, so history and staged-file
   security cannot yet be verified.
2. Raw/cache data rights and provider redistribution terms require owner review.
3. The workspace contains 6.3 GiB of ignored generated results and 31 files over
   10 MiB; force-adding or copying them would make the public release unsuitable.
4. The repository owner must confirm the right to apply the MIT license to all
   code selected for publication.

### Low

1. Ignored local work records contain user-specific absolute paths.
2. Generated HTML reports were excluded instead of exhaustively content-audited.
3. Dependency resolution is bounded but not locked.
4. GitHub Actions syntax was parsed as YAML but not validated with `actionlint`
   or a real workflow run.

### Informational

1. Provider credential names are referenced through environment variables; no
   literal values were found.
2. No risky serialized artifact or notebook was found outside excluded
   environments.

## 12. Files requiring manual inspection

Before selecting a first commit, manually inspect:

- every path reported by `git status --short` after `git init`;
- `.env.example`;
- `configs/universe_pit.yaml` and all historical configs selected for release;
- `LICENSE` and `CITATION.cff`;
- `reports/public/*.csv` and their source checksums;
- any generated HTML considered for inclusion;
- all `data/cache*` and `reports/output*` paths to confirm they remain ignored;
- legacy work records if the owner decides to publish any of them; and
- the complete staged diff.

## 13. Public-release recommendation

Within the performed working-tree scan, the proposed public source set has no
identified Critical or High severity secret finding. It is suitable for a
manual staged-set review, **not yet for immediate public push**.

Before publication, initialize or choose the intended Git repository, stage
only reviewed files, rerun `scripts/security_check.sh`, inspect the staged file
list and large files, confirm code/data licensing, and review any resulting Git
history.

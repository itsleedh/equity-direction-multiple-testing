# Public Release Checklist

## Research claims

- [x] Repository positioning centers on hypothesis testing and falsification.
- [x] The negative result is presented as the main research conclusion.
- [x] Accuracy, statistical significance, and profitability are distinguished.
- [x] Nominal and BH-FDR-adjusted results are distinguished.
- [x] The final headline uses the target-majority baseline.
- [x] The historical 442-combination denominator is qualified.
- [x] No result or performance value was guessed during restructuring.
- [ ] A human has rechecked every README claim against the named source.

## Code and tests

- [x] The pre-change canonical suite passed 80 tests.
- [x] The post-change canonical suite passed 86 tests in the existing local
  environment.
- [x] The synthetic smoke command passed in the existing local environment.
- [x] The final post-change suite passes in a clean environment.
- [x] The synthetic smoke command passes in a clean environment.
- [x] Package installation succeeds from `pyproject.toml`.
- [x] CI has been reviewed and run on GitHub.
- [ ] Optional ML backend differences have been reviewed.

## Data and licensing

- [x] Raw/cache market-data paths are ignored.
- [x] Generated multi-gigabyte result paths are ignored.
- [x] A deterministic fictional sample is documented.
- [x] The code license is separated from third-party data rights.
- [ ] Data-provider terms and redistribution restrictions have been reviewed by
  the repository owner.
- [x] No raw licensed market data is staged for commit.
- [ ] Every file selected for the first commit has been manually inspected.

## Security and privacy

- [x] A redacted working-tree secret scan was performed.
- [x] Both ripgrep and grep-fallback content-scan paths were exercised.
- [x] Polygon provider failures produce a credential-safe public traceback.
- [x] No notebook files were found.
- [x] No pickle, joblib, or model-checkpoint files were found outside `.venv`.
- [x] No credentials are present in the exact staged file set.
- [x] No pre-existing Git history existed before the audited first commit.
- [x] Large staged files have been reviewed.
- [x] Local paths and private identity details have been removed from staged
  files.

## Documentation and governance

- [x] Methodology, validation, results, limitations, and data policy are
  documented.
- [x] AI assistance is disclosed.
- [x] The security audit and restructuring report exist.
- [ ] Citation author metadata has been confirmed by the repository owner.
- [x] The repository name and public description have been selected.
- [ ] The user has manually reviewed `git diff` or an equivalent file-by-file
  comparison.
- [x] The user has approved commit.
- [x] The user has approved push.

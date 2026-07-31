# AI-Assisted Development Disclosure

Existing project records and retained task files indicate that AI tools,
including OpenAI Codex, were used for implementation assistance, refactoring,
review, documentation, and test development.

AI assistance was not treated as evidence that the research is correct.
In particular:

- research questions and acceptance criteria require human judgment;
- statistical controls and their applicability require human verification;
- experiment interpretation and final claims require manual review;
- AI-generated or AI-modified code is subject to automated tests and code
  review; and
- passing tests does not prove that a backtest is free from all leakage,
  selection bias, data error, or economic-modeling error.

The public-release restructuring was also AI-assisted. Its factual claims were
checked against code, configuration, local result summaries, and the test suite
where possible. Known evidence gaps—especially the historical 442-combination
denominator—are disclosed rather than filled with estimates.

Human reviewers should inspect the complete diff, validate data rights, confirm
the selected public files, and approve any future commit or push.

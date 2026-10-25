## Code Review Rules

### PR artifact cleanup

- Before requesting review and before merging, inspect the complete PR diff for temporary debugging or comparison helpers, one-off scripts, work logs, hard-coded local paths, and files named for the PR or issue rather than their continuing purpose.
- Remove artifacts needed only to prepare or review the PR. Keep reusable feature code, regression tests, specifications, operating instructions, and validation evidence; a PR or issue reference alone is not a reason to delete them.
- Organize permanent documentation by feature or concept so it remains understandable without the PR discussion. Keep development chronology and review receipts in the issue/PR or local notes, rather than production documentation or runtime code.
- Remove unused code, options, dependencies, and obsolete references; update links and check that the remaining documentation and commands refer to existing files. Summarize this cleanup in the PR checklist.
- Do not include or delete unrelated work or existing experimental artifacts during cleanup. Preserve the experimental data boundary below.

### GitHub Codex review requests

- Request a GitHub Codex review manually at most once per PR by default. Check the PR's existing Codex review history before posting `@codex review`; a new head does not reset the count.
- For any additional GitHub Codex review, explain the reason, the previously reviewed and current head SHAs, the findings addressed, and the current checks, then obtain explicit approval before posting another request. Local review and CI may continue without another GitHub request.

### Frame and time identity

- In repellent response analysis, keep each observation tied to its original frame and timestamp. Missing coordinates, invalid centers, and timestamp gaps must remain unavailable; do not compact rows, carry values across gaps, or calculate a direction or switching event across them. Preserve explicit source-frame mapping when saving or splitting a series.

### Primary rotation and stop measurements

- For V2 repellent-response output, treat the V2 sign, reference center, and angular velocity as the primary rotating series. Stop-candidate center and angular velocity are separate measurements: they must not fill missing primary values or enter primary CW rate and switching counts. If the required references or lever arm are unavailable, retain a missing stop measurement with its reason.
- AVI centroid coordinates use a downward-positive y axis. In this image space, a visually clockwise orbit has positive angular change; repellent-response output reports CW as positive for V2, validated, and legacy estimators.

### Experimental data boundary

- Do not commit or post measured coordinates, raw images or videos, reproducible measured time series, or generated plots derived from them. Keep those artifacts local. Use synthetic data and aggregate, non-reconstructable summaries for tests and public issue or PR evidence.

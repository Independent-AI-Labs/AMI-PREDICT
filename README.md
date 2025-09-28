# Predict Domain (Research)

Predict preserves our early experiments in trading and market-automation agents so future teams can mine the insights without reviving unsupported code paths. It captures what we learned about orchestrating financial workflows, even though the implementation predates current `/base` patterns.

## Current Guidance

- All legacy documentation and experiment logs now live under `research/`.
- Reuse any code here only after refactoring it to follow Base conventions
  (DataOps models, uv tooling, FastMCP contracts) and documenting the work in the
  active module docs.
- Tests and scripts in this package are not hooked into the orchestrator CI; treat
  them as experimental utilities.

When a trading or predictive workload is brought back to production, create fresh
modules and documentation aligned with the latest architecture instead of relying
on these research artifacts.

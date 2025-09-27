# Predict Domain (Research)

The Predict domain is retained as historical research exploring trading-focused
workflows. It does not conform to the current `/base` design patterns and is not
part of the supported orchestrator surface.

- All legacy documentation and experiment logs now live under `research/`.
- Reuse any code here only after refactoring it to follow Base conventions
  (DataOps models, uv tooling, FastMCP contracts) and documenting the work in the
  active module docs.
- Tests and scripts in this package are not hooked into the orchestrator CI; treat
  them as experimental utilities.

When a trading or predictive workload is brought back to production, create fresh
modules and documentation aligned with the latest architecture instead of relying
on these research artifacts.

# Contributing

## Workflow

1. Choose an unassigned workstream or issue with the team before starting.
2. Branch from the latest `main` using a short name such as `feature/data-loader` or `docs/experiment-plan`.
3. Make focused commits with imperative messages, for example `Add volume metadata validation`.
4. Push the branch and open a pull request that explains the change, validation performed, and any interface or dependency impact.
5. Request at least one teammate review. Resolve comments and keep the branch current with `main` before integration.
6. Merge only when checks and review pass. Do not push unfinished work directly to `main`.

Coordinate changes to shared interfaces in `src/` before merging. Never commit data, weights, credentials, or generated outputs. Record experiments with the template in `docs/EXPERIMENTS.md`.

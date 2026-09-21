# Contributing

## Workflow
Issue → branch → PR → 1 review → squash merge to `main`. No direct pushes to `main`.

## Branches
`feature/<issue#>-short-name`, `fix/<issue#>-short-name`, `experiment/<issue#>-short-name`
Keep branches short-lived (merge within 1-2 days).

## Commits
`feat: ...` `fix: ...` `docs: ...` `test: ...`

## Pull requests
- Link the issue (`Closes #N`)
- Reviewer must be someone other than the author
- Merge only when checks pass and comments are resolved

## Definition of Done
Works + reviewed + merged + docs updated if behavior changed.
ML work: also record dataset, metrics, and how it was evaluated.

## Secrets
Never commit secrets. Use `.env` (git-ignored) and keep `.env.example` updated.
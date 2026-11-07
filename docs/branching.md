# Branching

Two long-lived branches exist: `release` and `test`. The lab standard replaced the older `release`, `dev` and `test` policy that this page used to describe, and the scripts that enforced it.

| Branch | Holds |
| --- | --- |
| `release` | What is published. It moves only by a pull request from `test`, once the gates are green. |
| `test` | What is proven and waiting to be released. Every change reaches it by a pull request. |

Work happens on a short-lived branch named `xvc-<issue>/<type>/<slug>`, from an issue, into `test`. `CONTRIBUTING.md` is the whole path and `CLAUDE.md` holds the rules.

## What became of the old branches

GitHub's `dev` held one commit that `release` does not, the benchmark numbers in the README. It is kept as the tag `import/dev`. GitHub's `test` held nothing that `release` does not, so it needed no tag. Nothing was deleted.

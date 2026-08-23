# W01_TO_S14 mother trajectories

The three formal human mother trajectories below were collected with the
frozen implementation commit `98be459da6e1f24fd6688c4b535c8043d0853476`,
scene seed `0`, Pink IK, and an explicit 5-degree keyboard rotation step.
Every episode reports `SUCCEEDED`, `training_allowed=true`, three of three
fresh terminal votes, and a passing one-second drift hold.

| Mother | Episode ID | Actions | Strict Reader | Package SHA256 |
|---|---|---:|---|---|
| M01 | `w01-s14-train-m01-20260823-162935` | 255 | PASS | `1a8627aaae9d2a63515508006f7e73a9a3305b1b6cb287cda98c28230f99f7da` |
| M02 | `w01-s14-train-m02-20260823-181800` | 251 | PASS | `8ffba8ee64c9939a20c5165e5e9511ce20a99c7dd6fa284fb81079c24ad8b04d` |
| M03 | `w01-s14-train-m03-20260823-194629` | 267 | PASS | `a6c16d4b8fe609a060e656c7ca9fa4fb0f0d57711aaa047aa51737bf9990a200` |

The packages are stored outside Git under `/home/xyz/v2-packages/` and keep
their matching `.sha256` sidecars. Offline ground-truth details remain outside
the Canonical training episodes as required by the observation contract.

The repository default rotation step is changed to 5 degrees only after these
three frozen episodes. This later default-only change does not alter their
recorded actions or provenance.

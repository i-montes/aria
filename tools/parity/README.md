# Parity checks

Phase 2d's checks: `ARIA-SETUP.md` (the owner's Pi setup) walked item by item against Aria. The matrix with every
item, its check and its last result is `docs/parity.md`.

- `scratch-home.sh <command>` runs a command in Pi's exact session environment (`pi::session_env`, without
  `ANTHROPIC_*`) for a scratch copy of `~/.aria`, in a fresh git repo, then deletes the copy. It only reads `~/.aria`:
  the copy has its own agent directory, sessions and Engram data, so the installed dock and its memory are never
  touched. Its test is `scratch-home.test.sh`.
- `matrix.ts` and `matrix.test.ts` keep `docs/parity.md` honest: every setup item has a row, and every file and test a
  row names exists (`bun test tools/parity`).
- `smoke.sh` is the setup's smoke test: Aria's Pi answers "hola" short and in Spanish.

The evals that need real models live beside the extension they check, as in the owner's Pi:
`extensions/jev-router/eval.ts`, `extensions/sandbox/review-eval.ts` and `extensions/sandbox/accept.sh`. Run each
through the scratch home:

    sh tools/parity/scratch-home.sh bun $PWD/extensions/jev-router/eval.ts
    sh tools/parity/scratch-home.sh bun $PWD/extensions/sandbox/review-eval.ts
    sh tools/parity/scratch-home.sh bash $PWD/extensions/sandbox/accept.sh
    sh tools/parity/scratch-home.sh sh $PWD/tools/parity/smoke.sh

Jev (the classifier) is free. `accept.sh` and the smoke test use the daily model and spend a few cents of the Go quota.

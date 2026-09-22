# Contributing

Thanks for looking. This is a small extension with an unusually strict brief,
so most of what follows is about the brief rather than about style.

## The one rule everything else follows from

**Nothing may be specific to one machine.** This extension is installable by  
anyone, and the reference box it was written on two GPUs, ROCm, one of them driving a monitor is a sample of one. So:

- **No hardware constants in `src/`.** No GPU count, VRAM size, card index,
  backend name or model name. Ask the server: the module headers in `src/api/`
  say which endpoint answers what, and the server's own `/openapi.json` is the
  authority on payloads.
- **No recomputing what the server knows.** Memory figures come from
  `POST /api/inference/estimate-memory`, not from hand-rolled GGUF header
  parsing or KV-size tables. That code was deleted on purpose.
- Anything that cannot be answered portably is *shown and made correctable*, the wizard's display-GPU flag is the model for this rather than guessed at  
  silently.

Before opening a pull request, grep your diff for a number that is a fact about
your computer.

## The module headers are the contract

Every file in `src/` opens with a comment saying what it is for, which rule it  
implements and usually which bug taught us the rule. That is where the design lives. `src/hardware/budget.ts` carries the fit rule and why it splits by
card size rather than by budget; `src/endpoint.ts` carries the locality rule and
the day it was needed; `src/api/lifecycle.ts` carries the three shapes a refusal
arrives in.

**If the behaviour changes, change the header in the same pass as the code.** A
patch that makes a screen behave differently from the comment above it is
incomplete, however good the code is. 

## What "done" means here

A claim is made only when it was **observed**. "Verified on the reference box"  
means someone ran it and watched; anything that needs a real GPU load, or a  
second machine, or a topology this hardware cannot produce, stays *unclaimed* and says why. An honest "not tested on CUDA" is a perfectly good contribution, a tick that was never run is not.

`npm run check` — typecheck plus the test suite — must be green before you send
anything.

## Development

```bash
npm install
npm run check      # tsc --noEmit, then node --test over the sources
```

TypeScript, loaded by Pi through jiti. **There is no build step**, which has
two consequences worth knowing before you spend an afternoon on them:

- **No constructor parameter properties, and no `enum`.** `node --test` strips
  types rather than compiling them, and those two constructs cannot be stripped.
  They typecheck and they load under jiti, so the failure appears only in the
  tests, and only as a confusing one.
- **Import pi-ai only as `@earendil-works/pi-ai` or `…/pi-ai/compat`.** Deeper
  subpaths typecheck but do not resolve inside Pi's runtime, and the extension
  then fails to load *silently*.

Pi's own packages (`pi-ai`, `pi-coding-agent`, `pi-tui`, `typebox`) belong in
`peerDependencies` at `"*"` and are never bundled — and are pinned in
`devDependencies` to a real Pi version, because `"*"` resolves to whatever npm
has today and its types drift from the runtime Pi actually loads.

Use `getAgentDir()` for config paths.

## Testing a change to a screen

Unit tests cover the reducers — a keystroke in, a state out — and they are
where a new key binding or a new field belongs. They are also not enough on
their own: every UI bug that mattered in this project (a footer naming no
model, "0.0 GiB freed", every GPU tagged `display`, a progress bar showing the
previous load's 100 %) passed its unit tests and was obvious within one real
session.

So UI work is also driven for real, against an **isolated profile**
(`PI_CODING_AGENT_DIR` pointing at a temp directory with a copy of `auth.json`,
then `pi install .`) so your own configuration is never the thing under test.
Driving it from a pty script rather than by hand makes the run repeatable; if
you do, note that Pi negotiates the Kitty keyboard protocol at startup, so
`ctrl+alt+…` has to be sent in Kitty form (`\x1b[117;7u` for `ctrl+alt+u`)
while plain keys and arrows go through unencoded.

**Put the GPUs back.** If your test loaded or unloaded anything, restore what
was resident before you started and check the `llama-server` argv matches flag
for flag — `--port` is ephemeral and always differs. Someone works on that
machine.

## Things to be careful with

- **Never OOM-probe.** The estimator is free and allocates nothing; binary-search
  that instead. Load for real only at a value already believed to fit, and ask
  twice when the display GPU is involved — a failed load there freezes a desktop.
- **Guard every overlay with `ctx.mode === "tui"`**, and answer in one line
  instead. `pi -p` must keep working in a pipeline.
- **No background work in the extension factory** — start it in `session_start`,
  clear it in `session_shutdown`, and keep that handler idempotent.
- **No secret in a log, a status line or an error.** The key reaches exactly one
  place: an `Authorization` header.
- **Errors carry the server's own words** when it gave any.

These are standing gates, not one-time chores: every new screen and every new
call is checked against them again, because the fifth overlay is exactly where
the `ctx.mode` guard gets forgotten.

## Platform claims

Only Linux with ROCm has been run end to end. CUDA, macOS and Windows are
marked experimental or unsupported in the README's platform table, and the
honest thing to do with that table is to move a row **because you ran it**, in
the same pull request as whatever made it work. Reports from other hardware are
genuinely useful even with no patch attached.

## Submitting

Issues and pull requests go to
[github.com/TheLoomLabs/pi-unsloth](https://github.com/TheLoomLabs/pi-unsloth)
— [issues](https://github.com/TheLoomLabs/pi-unsloth/issues),
[pull requests](https://github.com/TheLoomLabs/pi-unsloth/pulls).

Keep a pull request to one milestone box, or one bug, where you can. Say what
you observed, not only what you changed — this project's history is a chain of
"verified on …" notes, and yours belongs in it.

By contributing, you agree that your work is licensed under the MIT terms in
[`LICENSE`](LICENSE).

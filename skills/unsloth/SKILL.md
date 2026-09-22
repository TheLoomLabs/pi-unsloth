---
name: unsloth
description: Drive the Unsloth Studio server behind Pi's `unsloth` provider — report what is loaded and what the GPUs hold, list models and resolve their paths, price a context with the server's own memory estimator, decide whether a configuration fits, pre-flight and run a load, and free the cards. Use when the user asks what is on the GPUs, whether a model or a context length will fit, to load, switch or unload a local model, or to diagnose an Unsloth server that is down, unauthorized or missing models.
license: MIT
compatibility: Needs `curl` and `node`, and an Unsloth Studio server this machine can reach (local or over the LAN).
---

# Unsloth Studio

Unsloth Studio runs the local models Pi talks to. This skill is how **you**
ask it questions and, with the user's say-so, change what it is holding.

Everything below goes through one script, which resolves the endpoint and the
key the way the extension does and never prints the key:

```bash
scripts/unsloth-api GET  /api/system
scripts/unsloth-api POST /api/inference/estimate-memory '{"model_path":"…"}'
scripts/unsloth-api PUT  /api/settings/openai-auto-switch/overrides @payload.json
```

`scripts/` is next to this file — resolve it against this skill's own
directory, not against the working directory, which is the user's project and
usually somewhere else entirely.

The body goes to stdout, `HTTP <status> <method> <path>` to stderr, and the
exit status is 0 only for 2xx. Keep the status: on this API it is half the
answer.

## First: is this yours to do?

The extension already has screens for the things a user would rather do
themselves, and those screens write settings correctly, show the cost of a
choice and ask before spending VRAM. Point at them instead of re-implementing
them:

| The user wants | Say this, don't do it |
|---|---|
| To see the panel, the GPUs, what is loaded | `ctrl+alt+u`, or `/unsloth` |
| To size or re-tune a model | `/unsloth add <model>` — the sizer finds max safe context by binary search |
| Sampling defaults for a model | `/unsloth sampling <model>` |
| To re-run hardware detection, or point at another server | `/unsloth setup` |
| **To free the GPUs** | `/unsloth off`, or `ctrl+alt+o` |
| One line about the server, in a script or `pi -p` | `/unsloth status` |

Use this skill for the rest: answering questions, diagnosing, and doing the
steps when the user has asked *you* to do them.

## Rules

These are the project's own invariants. They are not style preferences — each
one is here because ignoring it costs someone their session or their desktop.

- **Never OOM-probe.** Never load a configuration to find out whether it fits.
  `estimate-memory` allocates nothing and is free to call in a loop; that is
  what "finding the limit" means here.
- **`available: true` is not a fit check.** It permits CPU offload. Apply the
  fit rule below yourself — verified again today: a 32 K estimate on the 27B
  answers `available: true` with `gpu_bytes` 32.7 GB against a 23.98 GB card.
- **Ask before you change what the cards hold.** A load or an unload interrupts
  whatever the user — or another Pi window — is doing with that server. Report
  first, act second.
- **Extra care with the display GPU.** A failed load on the card driving the
  monitor can freeze the desktop. `~/.pi/agent/unsloth.json` says which card
  that is (`"display": true`).
- **Unloading does not free VRAM.** `POST /api/inference/unload` clears the
  server's state and leaves the weights resident; only terminating
  `llama-server` gives the memory back. That second half is `/unsloth off`'s
  job — it checks the endpoint is local first. Do not go hunting for the
  process yourself.
- **A remote endpoint is not this machine.** If `baseUrl` is not loopback and
  is not one of this machine's own addresses, there is no autostart, no process
  to kill and no `/sys/class/drm` to read. Say what could not be done rather
  than doing it to the wrong computer.
- **VRAM is not granted or released synchronously.** `/load` answers — with
  `load-progress` already `ready` — seconds before the cards report the
  weights, and an unload's memory comes back in plateaus. Poll until the figure
  stops moving, and say *nothing* rather than zero when it never moved.
- **Never print the API key**, and never paste it into a command line.

## Recipes

### What is the server doing?

```bash
scripts/unsloth-api GET /api/health            # up at all? also carries hf_endpoint and version
scripts/unsloth-api GET /api/system            # .gpu.devices — topology plus live VRAM
scripts/unsloth-api GET /api/inference/loaded-models
scripts/unsloth-api GET /api/inference/status  # capabilities of the loaded model only
```

In `/api/system`, a device with `unified_memory` or `shared_memory` true is an
iGPU — drop it, it is not a compute GPU. `index` is the integer every other
endpoint means by `gpu_ids`; **never** map it to a `/sys/class/drm/cardN`
number, which on the reference box is inverted.

A 401 means the key is wrong: the user runs `/login`. A connection refused
means nothing is listening — the extension starts Studio itself on the next
session unless `UNSLOTH_AUTOSTART=off`.

### Which models are there, and where?

```bash
scripts/unsloth-api GET /api/inference/models  # the catalogue: downloaded models, loaded or not
scripts/unsloth-api GET /api/models/local      # the filesystem view — this is where model_path comes from
```

`estimate-memory`, `validate` and `load` all want a **`model_path`** from
`/api/models/local`, not the display name. If a model the user swears is on
disk is missing, check `GET /api/models/scan-folders` before saying it is gone:
`models_dir` is resolved against the server's working directory, so a folder
that must be found regardless of cwd has to be registered
(`POST /api/models/scan-folders {"path": "/absolute/path"}`).

### What will this configuration cost?

```bash
scripts/unsloth-api POST /api/inference/estimate-memory '{
  "model_path": "/path/to/model", "gguf_variant": "Q8_0",
  "n_ctx": 32768, "cache_type_kv": "q8_0", "n_parallel": 1,
  "tensor_parallel": false, "selected_gpu_ids": [1]
}'
```

- Omit `n_ctx` (or send `0`) to price the model's **native** context and read
  the ceiling back out of the echoed `n_ctx`. Free, no load.
- **Send `n_parallel` explicitly.** Left out, the server prices its own default
  of 4 — worth ~1.4 GiB on the reference 27B, which is the difference between
  fitting and not.
- `kv_estimable: false` means the KV size is *unknown*, not zero, and
  `total_bytes` is then a lower bound. Say so; do not size against it.
- `available: false` carries a code, not a sentence: `not_gguf`,
  `not_downloaded`, `unsupported_source`, `unsizable`.

### Does it fit?

Budgets come from `~/.pi/agent/unsloth.json`, which the wizard wrote:

```
budget[g] = totalGiB − idleUsedGiB − headroomGiB          (3 GiB on the display card, 0.5 headless)

single GPU        →  gpu_bytes ≤ budget[g]
tensor-parallel   →  for each g: gpu_bytes × share[g] ≤ budget[g]
                     share[g] = that card's share of total VRAM, because that is
                     what llama.cpp does — there is no tensor_split to send
```

Placement, in order: a headless card that fits → the display card → several
cards tensor-parallel → nothing fits, and then report the **smallest**
shortfall, because "over by 2.3 GiB" is a number the user can act on.

### Pre-flight, then load

```bash
scripts/unsloth-api POST /api/inference/validate '{"model_path":"…","max_seq_length":32768,"gpu_ids":[1],"tensor_parallel":false,"cache_type_kv":"q8_0","n_parallel":1}'
```

Read the answer by its shape — this is why the script prints the status:

| Answer | Means | Do |
|---|---|---|
| `200` + `valid: false` | the server refused the configuration | refuse; show `message` |
| `400` + `{"detail": "<string>"}` | an application refusal (`Invalid gpu_ids [7]: …`) | refuse; show the string |
| `422` + `{"detail": [ … ]}` | FastAPI rejecting **our payload shape**, not the user's config | the pre-flight could not run; do not call it a refusal |
| `404` / `5xx` / silence | server too old, or down | say so |

`valid: true` is not "this will load" — it answers whether the identifier
resolves. Per-GPU fit is still your question.

Then, only with the user's agreement:

```bash
scripts/unsloth-api POST /api/inference/load '{"model_path":"…","gguf_variant":"Q8_0","custom_context_length":32768,"kv_cache_dtype":"q8_0","tensor_parallel":false,"gpu_ids":[1],"n_parallel":1}'
scripts/unsloth-api GET  /api/inference/load-progress   # phase, fraction — poll until ready or an error
```

Prefer loading a model **on its stored settings**: read them from
`GET /api/settings/openai-auto-switch/overrides` (keyed `"<model_id|path>:<QUANT>"`)
and build the load body from that entry. Hand-built settings discard the tuning
the sizer measured — and auto-switch silently drops speculative decoding, which
is why an explicit load exists at all.

Writing an override is a `PUT` of **one** model, `model_id` carrying the whole
composite key, and it **replaces** that model's entry — merge over the stored
one or you will erase the fields you did not send. `/unsloth add` does this
properly; prefer it.

## When the answer is "I don't know"

Say that. The honest sentence — "the server did not answer `/api/system`, so I
cannot tell you what the GPUs hold" — is worth more than a guess, and it is the
tone the rest of this extension keeps. Three questions this API genuinely
cannot answer: a model's preferred sampling (request-only; that is
`/unsloth sampling`), per-model capabilities for anything not currently loaded,
and whether the weights have actually left VRAM after an unload on a remote
server.

## Reference

**The server's own `/openapi.json` is the authority on payloads — fetch it
rather than guessing a field name.** What it will not tell you is everything in
the Rules section above: which answers are honest, which are lower bounds, and
which endpoints do less than their names suggest. That part was established
against a live server and is written down here because the spec cannot say it.

`GET /api/health` carries the server version and its `hf_endpoint`; if something
below does not exist on the server in front of you, that is the first thing to
check.

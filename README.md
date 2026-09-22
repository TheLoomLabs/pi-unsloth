# pi-unsloth

A [Pi](https://pi.dev) extension that turns an [Unsloth Studio](https://unsloth.ai) server on this machine or on the GPU box across the room into a first-class Pi provider.

Every model you have downloaded shows up in `/model` on its own. The server
starts when a session needs it. Each model loads with *its own* tuned context,  
placement and speculative-decoding settings, re-applied on every switch. A status line keeps live VRAM in front of you, a panel loads and unloads without leaving the editor, and a sizer works out what a new model can hold on the hardware it actually finds by asking the server's memory estimator, not by guessing from a table.

Nothing is hardcoded about anyone's hardware: no GPU count, no VRAM size, no
card index, no model name.

## Requirements


|                |                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------- |
| Pi             | 0.85.1 or later                                                                           |
| Node           | 20 or later                                                                               |
| Unsloth Studio | any version to list, load and unload · **2026.9 or later** for sizing                     |
| Server OS      | Linux with ROCm is the verified configuration — see [Platform support](#platform-support) |


Sizing is gated on the server version rather than faked: an older server
disables it with a message naming what it needs, and everything else keeps
working.

## Install

```bash
pi install git:github.com/TheLoomLabs/pi-unsloth
```

Or from a clone, which is also how you work on it:

```bash
git clone https://github.com/TheLoomLabs/pi-unsloth.git
pi install ./pi-unsloth
```

Then start Pi and give it the server's API key:

```
/login → Unsloth → paste the key
```

The key can also come from `UNSLOTH_API_KEY`. `/api/health` answers without
one, so a wrong key is reported as a rejected key rather than as a dead server.

## Quick start

1. `pi` — the extension starts Unsloth Studio if it is not already up, and the
 editor stays usable while it comes up.
2. `/unsloth setup` — once per machine, when you are ready: it lists the GPUs
 the server reports, says which one it thinks drives your monitor **and why**,
 and lets you overrule it. Nothing is written until you press `⏎`. Until you
 run it, the first session on a machine says so in one line and opens nothing.
3. `/model` — pick any downloaded model. It loads with its tuned settings.
4. `ctrl+alt+u` — the panel: live VRAM bars, load, unload, size.
5. `ctrl+alt+o` — free the GPUs when you are done.

## What it looks like

**The status line** (`ctrl+alt+f` hides it, and hiding it stops the 4 s poll
too):

```
 ⬢ Qwen3.8-27B · 192K · ▓▓▓▓▓▓▓░ 39.8/48.0 GiB      loaded
 ⬢ Qwen3.8-27B · 192K · ▓▓▓▓▓▓▓░ 39.8/48.0 GiB ▸    generating
 ◐ loading Qwen3.8-27B · 68%                        loading
 ○ unsloth idle · ░░░░░░░░ 1.6/48.0 GiB             server up, nothing loaded
 ◌ unsloth offline                                  nothing is listening
```

**The panel** — `ctrl+alt+u` or `/unsloth`:

```
╭─ Unsloth  39.7/48.0 GiB ───────────────────────────────────────────────╮
│   GPU 0  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░  21.3/24.0  display                │
│   GPU 1  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░  18.4/24.0                         │
│                                                                        │
│ → ● Qwen3.8-27B    Q8_0     192K  tp 0+1                               │
│   ○ Qwen3-4B-GGUF  Q4_K_M    40K  gpu 1                                │
│   ○ gemma-3-27b-it Q4_K_M      —  not sized                            │
│   ○ Llama-3.3-70B  Q4_K_M      —  won't fit                            │
│                                                                        │
│   ↑↓ select  ⏎ load  a size  s sampling  u unload  r refresh  esc      │
╰────────────────────────────────────────────────────────────────────────╯
```

The bars are the machine's own figures - they agree with`/sys/class/drm/*/device/mem_info_vram_used` to the decimal and integrated graphics are excluded from both the bars and the total.

**The sizer** - `a` on a row, or `/unsloth add <model>`:

```
╭─ Size  Qwen3.8-27B  Q8_0 ──────────────────────────────────────────────╮
│   weights      27.9 GiB    compute       2.2 GiB                       │
│   kv @ 196608   9.4 GiB    drafter       1.4 GiB                       │
│   projector     0.3 GiB                                                │
│                                                                        │
│   total        38.3 GiB                                                │
│   GPU 0        19.2 GiB of 19.4 GiB budget                             │
│   GPU 1        19.2 GiB of 23.4 GiB budget                             │
│   fits ✓                                                               │
│                                                                        │
│   context     ‹ 196608 ›  max safe 204800                              │
│   kv dtype    ‹ q8_0 ›                                                 │
│   placement   ‹ tensor-parallel 0+1 ›                                  │
│   speculative ‹ mtp ›                                                  │
│   slots       ‹ 1 ›                                                    │
│                                                                        │
│   ↑↓ field  ←→ adjust  ⏎ apply  v verify  esc                          │
╰────────────────────────────────────────────────────────────────────────╯
```

**The setup wizard** -`/unsloth setup`:

```
╭─ Unsloth setup ────────────────────────────────────────────────────────╮
│   ───────────────────────────────────────────────                      │
│     /\        /\                                                       │
│    (  )      (  )     █ █ █▄█ █▀▀ █   █▀█ ▀█▀ █ █                      │
│     \  \____/  /      █ █ █ █ ▀▀█ █   █ █  █  █▀█                      │
│      \ (o  o) /       ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀ ▀▀▀  ▀  ▀ ▀                      │
│        \ ~~ /                                                          │
│                                                                        │
│   Server    127.0.0.1:8888  ✓ up  v2026.9.7  rocm        e change      │
│                                                                        │
│   Detected 2 compute GPUs  (1 integrated, ignored)                     │
│                                                                        │
│ → GPU 0  AMD Radeon Graphics  24.0 GiB                                 │
│       1.60 GiB in use at idle                                          │
│       ▸ display attached — card1-DP-3 connected                        │
│       headroom  ‹ 3.0 GiB ›  19.4 GiB usable                           │
│                                                                        │
│   GPU 1  AMD Radeon Graphics  24.0 GiB                                 │
│       0.03 GiB in use at idle                                          │
│       ▸ headless — preferred for models                                │
│       headroom  ‹ 0.5 GiB ›  23.4 GiB usable                           │
│                                                                        │
│   Wrong?  d  toggles the display flag on the selected GPU              │
│                                                                        │
│   ↑↓ move  ←→ adjust  d toggle  e server  ⏎ save  esc cancel           │
╰────────────────────────────────────────────────────────────────────────╯
```

## Commands and keys


| Key          | Action                       |
| ------------ | ---------------------------- |
| `ctrl+alt+u` | Open the panel               |
| `ctrl+alt+o` | Unload and free the GPUs     |
| `ctrl+alt+f` | Show or hide the status line |



| Command                     | Action                                                    |
| --------------------------- | --------------------------------------------------------- |
| `/unsloth`                  | The panel                                                 |
| `/unsloth add <model>`      | Size a model against this hardware                        |
| `/unsloth sampling <model>` | Set a model's sampling defaults                           |
| `/unsloth setup`            | Run the setup wizard — the only thing that opens it       |
| `/unsloth footer [on|off]`  | Show or hide the status line, and remember it             |
| `/unsloth off`              | Unload                                                    |
| `/unsloth status`           | One line about the server — for scripts and non-TUI modes |


Every shortcut is re-bindable through `~/.pi/agent/keybindings.json`. The
extension registers no single-modifier shortcuts, so it cannot shadow a Pi
default. Below 50 columns, and in non-TUI modes such as `pi -p`, the overlays
are not drawn at all and the commands answer in one line instead.

### Letting the model drive it

The package also installs a skill, so Pi's own model can answer questions about
the server instead of sending you to a screen: what the GPUs are holding, which
models are on disk and where, whether a context length will fit, what a failed
load actually said. It talks to Unsloth through one helper
(`skills/unsloth/scripts/unsloth-api`), which resolves the same endpoint and
key the extension does and never prints the key.

It is deliberately lopsided: the model reports and diagnoses freely, and for anything that spends VRAM or writes settings it points at the command that does it properly -`/unsloth add`, `/unsloth sampling`, `/unsloth off` — because
those ask first. `/skill:unsloth` loads it on demand; `pi config` turns it off.

## Sizing a model

`a` in the panel, or `/unsloth add <model>`, opens the sizer on the model's
**stored** configuration - an override you tuned by hand is what you came to adjust, not something to discard.

- Every figure comes from `POST /api/inference/estimate-memory`. The estimator
allocates nothing, so the whole screen is free: `max safe` is found by binary
search over the context length, not by loading anything.
- The fit verdict is **per GPU**, against each card's own budget. The
estimator's `available: true` permits CPU offload and is not a fit check  the reference 27B prices at 30.4 GiB with `available: true` on a card where
this screen correctly says `over by 13.8 GiB`.
- `⏎` writes the model's entry in Unsloth's auto-switch overrides, preserving
every other model's settings and the rest of this model's own.
- `v` verifies for real: one load, one measurement, one unload, and the
difference between the estimate and what the cards actually held is stored as
a per-machine correction. There is no OOM-probing loop anywhere in this
extension, and a verification on the GPU driving your monitor asks a second
time before it starts.
- Every load  from the panel, from a model switch, from the sizer  goes through `POST /api/inference/validate` first, so a bad configuration is
refused in the server's own words without touching a GPU.

## Sampling defaults

`s` in the panel, or `/unsloth sampling <model>`:

```
╭─ Sampling  Qwen3-4B-GGUF ──────────────────────────────────────────────╮
│   Sent with every request Pi makes to this model.                      │
│                                                                        │
│   temperature ‹ 0.60 ›  0–2  step 0.05                                 │
│   top_p       ‹ 0.95 ›                                                 │
│   top_k       ‹ 20 ›                                                   │
│   min_p       ‹ default ›  the server decides                          │
│                                                                        │
│   from Qwen/Qwen3-4B — adjust to make them yours                       │
│                                                                        │
│   ↑↓ field  ←→ adjust  f model card  x clear  ⏎ save  esc              │
╰────────────────────────────────────────────────────────────────────────╯
```

This is the one setting nothing on the machine can work out for you: Unsloth's
API carries `temperature` and its three neighbours only as things a client
*sends*, never as something it reports. So there are two ways to fill the
screen in, and it always says which one you are looking at:

- `**f**` reads the model's own `generation_config.json` from Hugging Face —
where Qwen's `0.6 / 0.95 / 20` actually comes from. GGUF repos don't carry
that file, so it follows the repo's `base_model` once to the model it was
quantised from. A gated repo is reported as gated (set `HF_TOKEN` if you have
access), a model that is a plain file on disk is answered without a request,
and `PI_OFFLINE` skips the whole thing.
- `**←→**` is you. The first adjustment makes the values yours, and the line
under them stops crediting the model card.

`default` means the sampler is **not sent at all**, so the server's own value
stands — which is different from sending today's default as a number. `x`
unpins one; unpinning the last one removes the model's entry.

Saved into `~/.pi/agent/unsloth.json` and published to Pi as the model's
`samplingParams` on the next catalogue refresh, which happens immediately.

## Coming from a hand-written `models.json`

If you already list Unsloth models in `~/.pi/agent/models.json`, delete that
provider block once this is installed. Pi composes `models.json` **above** a
registered provider, so a `contextWindow` typed there shadows the one the
extension just sized — the number changes on disk and not in `/model`.

Everything that block was doing is derived now: the catalogue from the server,
the context from the model's override, the capabilities from the model itself,
and the sampling from the screen above. Nothing in this extension reads or
writes `models.json`; the published catalogue lives in Pi's own
`models-store.json`, which is also what keeps `/model` populated when the
server is down.

## Two machines

The GPU box and the machine you type on do not have to be the same computer.  
Unsloth serves everything this extension needs over HTTP  the catalogue, the topology, the estimator, load and unload, and the inference itself  so a laptop with no GPU can drive a server across the room.

Point at it with `/unsloth setup` → `e`, and type `host:port`, a bare host
(8888 assumed) or a full URL. `⏎` probes `/api/health` before accepting and
reports the real cause in place if it fails.

Three things have to be true on the server side:

- **Bind address.** Unsloth Studio starts on `127.0.0.1` by default. Start it
with `-H 0.0.0.0` and let the port through the firewall, or nothing off-box
can reach it.
- **The key is the server's.** Run `/login` on the client and paste the key
*that* server accepts.
- **It is plain HTTP.** The bearer token travels in a header in the clear. That is fine on a trusted LAN and not fine over anything wider  put a TLS proxy in front and give the wizard the full `https://` URL.

Three things behave differently against a remote endpoint, and the extension
says so rather than pretending:


|                     | Remote behaviour                                                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Autostart           | Off. Starting a local server would bind a different server than the one you configured                                                                                                                        |
| Freeing VRAM        | API unload only. There is no endpoint that stops the inference process, so the weights stay resident until someone acts on that machine — and `/unsloth off` names that host instead of saying "this machine" |
| Display cross-check | Off. `/sys/class/drm` describes the client, not the server. Idle VRAM still decides, and `d` still overrules it                                                                                               |


No local process is ever enumerated or signalled when the endpoint is not this
machine. "This machine" means loopback, `localhost`, or an address that belongs to one of this machine's own interfaces  so typing your own LAN address does not cost you the ability to free your own VRAM.

## Configuration

Everything resolves **environment → profile → default**, so any setting can be
overridden for a single run.


| Variable                      | What it does                          | Default                                               |
| ----------------------------- | ------------------------------------- | ----------------------------------------------------- |
| `UNSLOTH_BASE_URL`            | Server address                        | the profile's `baseUrl`, else `http://127.0.0.1:8888` |
| `UNSLOTH_API_KEY`             | API key                               | Pi's credential store (`/login`)                      |
| `UNSLOTH_AUTOSTART`           | May the extension start Studio itself | `on`                                                  |
| `UNSLOTH_AUTO_UNLOAD_ON_EXIT` | Free the GPUs when Pi quits           | `off`                                                 |
| `UNSLOTH_FOOTER`              | Draw the status line                  | `on`                                                  |
| `UNSLOTH_LAUNCH_COMMAND`      | How to start Studio                   | `unsloth-studio`                                      |
| `HF_TOKEN`                    | Read a gated model's card with `f`    | unset — sent as a header, never stored or printed     |
| `PI_OFFLINE`                  | Pi's own switch; `f` makes no request | unset                                                 |


`autoUnloadOnExit` is off by default because a second Pi window  or anything else using that server would lose its model without having asked for it. It also fires only on a real quit, never on `/new` or `/reload`.

The wizard writes `~/.pi/agent/unsloth.json`. It is documented, hand-editable,
written atomically, and unknown keys survive a rewrite untouched:

```jsonc
{
  "version": 1,
  "baseUrl": "http://127.0.0.1:8888",
  "backend": "rocm",
  "unslothVersion": "2026.9.7",
  "gpus": [
    { "index": 0, "name": "AMD Radeon Graphics", "totalGiB": 23.98,
      "idleUsedGiB": 1.21, "display": true,  "headroomGiB": 3.0,
      "displayEvidence": ["idle-vram", "drm-connector:card1-DP-3"] },
    { "index": 1, "name": "AMD Radeon Graphics", "totalGiB": 23.98,
      "idleUsedGiB": 0.03, "display": false, "headroomGiB": 0.5,
      "displayEvidence": [] }
  ],
  "calibration": { "estimateDeltaGiB": 0.62, "samples": 2,
                   "unslothVersion": "2026.9.7", "backend": "rocm" },
  "policy": { "preferHeadless": true, "autoUnloadOnExit": false,
              "footer": true, "kvDtype": "q8_0", "ctxStepTokens": 4096 },
  "sampling": {
    "ggml-org/Qwen3-4B-GGUF": {
      "params": { "temperature": 0.6, "top_p": 0.95, "top_k": 20 },
      "source": "hub", "from": "Qwen/Qwen3-4B"
    }
  }
}
```

`headroomGiB` is the one number worth tuning: VRAM left free on each card, 3 GiB by default on the GPU driving a display and 0.5 GiB on a headless one. If you point the client at a different server, the GPU data is discarded and re-detected, a headroom set for a 24 GiB card silently applied to a 12 GiB one is exactly how an OOM happens  while your policies and calibration are kept.

A stored calibration is dropped, not reused, when the server's Unsloth version
or backend changes.

## Platform support

Honest version: one configuration has been run end to end, and the rest is
marked by what has actually been exercised.


| Configuration                      | State                                                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Server: Linux + ROCm**           | **Verified** end to end on a 2 × 24 GiB box (plus an integrated GPU, correctly ignored)                                                               |
| Server: Linux + CUDA               | **Experimental.** Topology, estimates and the DRM cross-check all come from the same code paths, so it is expected to work — nobody has run it        |
| Server: macOS                      | **Experimental.** No sysfs cross-check, so the display GPU rests on idle VRAM and your correction; freeing VRAM shells out to `ps`, which is untested |
| Server: Windows                    | **Unsupported.** Process enumeration has no implementation there, so an unload cannot free VRAM — it says so instead of claiming success              |
| **Client on any OS, Linux server** | The weakest claim and the most likely to work: a client makes no sysfs reads and enumerates no processes, so only the HTTP path is exercised          |


## Development

```bash
npm install
npm run check      # typecheck + tests
```

TypeScript, loaded by Pi through jiti  no build step. Patches are welcome: [`CONTRIBUTING.md`](CONTRIBUTING.md) has the invariants, what "done" means here,
and the two traps that only show up at runtime.

The source, issues and pull requests live at
[github.com/TheLoomLabs/pi-unsloth](https://github.com/TheLoomLabs/pi-unsloth), a report from hardware this was not written on is useful even with no patch attached.

## Licence

MIT — see [`LICENSE`](LICENSE).
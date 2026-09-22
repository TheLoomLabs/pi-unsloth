import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { visibleWidth } from "@earendil-works/pi-tui";

import { DISPLAY_HEADROOM_GB, HEADLESS_HEADROOM_GB } from "../src/hardware/budget.ts";
import { PROFILE_VERSION, type MachineProfile } from "../src/hardware/profile.ts";
import {
  HEADROOM_STEP_GB,
  applyKey,
  conclusion,
  countLine,
  endpointLines,
  profileFrom,
  serverLine,
  usable,
  usageLine,
  wizardLines,
  wizardSummary,
  type WizardGpu,
  type WizardState,
  type WizardTheme,
} from "../src/ui/wizard.ts";

/** Themeless theme: the tests are about the words, not the colours. */
const theme: WizardTheme = { fg: (_token, text) => text, bold: (text) => text };

function gpu(overrides: Partial<WizardGpu> = {}): WizardGpu {
  return {
    index: 0,
    name: "AMD Radeon Graphics",
    totalGb: 23.98,
    usedGb: 1.21,
    idleUsedGb: 1.21,
    display: true,
    headroomGb: DISPLAY_HEADROOM_GB,
    evidence: ["idle-vram", "drm-connector:card1-DP-3"],
    connected: ["card1-DP-3"],
    touched: false,
    ...overrides,
  };
}

function state(overrides: Partial<WizardState> = {}): WizardState {
  return {
    server: {
      baseUrl: "http://127.0.0.1:8888",
      up: true,
      detail: undefined,
      version: "2026.9.7",
      backend: "rocm",
      remote: false,
    },
    gpus: [
      gpu(),
      gpu({
        index: 1,
        usedGb: 0.03,
        idleUsedGb: 0.03,
        display: false,
        headroomGb: HEADLESS_HEADROOM_GB,
        evidence: [],
        connected: [],
      }),
    ],
    ignoredCount: 1,
    displaySource: "idle-vram",
    sizing: { enabled: true, reason: undefined, version: "2026.9.7" },
    degradedReason: undefined,
    connectorNote: undefined,
    selected: 0,
    changed: false,
    editing: undefined,
    editError: undefined,
    probing: false,
    ...overrides,
  };
}

const ESC = "\u001b";
const ENTER = "\r";
const UP = "\u001b[A";
const DOWN = "\u001b[B";
const LEFT = "\u001b[D";
const RIGHT = "\u001b[C";

describe("applyKey", () => {
  it("writes nothing on escape", () => {
    assert.equal(applyKey(state(), ESC).outcome, "cancel");
  });

  it("saves on enter", () => {
    assert.equal(applyKey(state(), ENTER).outcome, "save");
  });

  it("has nothing to save when there is no topology, so enter closes", () => {
    assert.equal(applyKey(state({ gpus: [], degradedReason: "the server did not answer" }), ENTER).outcome, "cancel");
  });

  it("moves the selection, and stops at both ends", () => {
    assert.equal(applyKey(state(), UP).state.selected, 0);
    assert.equal(applyKey(state(), DOWN).state.selected, 1);
    assert.equal(applyKey(state({ selected: 1 }), DOWN).state.selected, 1);
    assert.equal(applyKey(state({ selected: 1 }), UP).state.selected, 0);
  });

  it("adjusts the selected GPU's headroom, and only that one's", () => {
    const { state: next } = applyKey(state(), RIGHT);
    assert.equal(next.gpus[0]?.headroomGb, DISPLAY_HEADROOM_GB + HEADROOM_STEP_GB);
    assert.equal(next.gpus[1]?.headroomGb, HEADLESS_HEADROOM_GB);
    assert.equal(next.changed, true);
  });

  it("keeps headroom inside the card, and free of floating-point dust", () => {
    let current = state({ gpus: [gpu({ headroomGb: 0.5 })] });
    for (let press = 0; press < 3; press++) current = applyKey(current, LEFT).state;
    assert.equal(current.gpus[0]?.headroomGb, 0);
    let up = state({ gpus: [gpu({ totalGb: 2, headroomGb: 1.8 })] });
    for (let press = 0; press < 3; press++) up = applyKey(up, RIGHT).state;
    assert.equal(up.gpus[0]?.headroomGb, 2);
  });

  it("`d` overturns the display flag and says the user did it", () => {
    const { state: next } = applyKey(state({ selected: 1 }), "d");
    assert.equal(next.gpus[1]?.display, true);
    assert.deepEqual(next.gpus[1]?.evidence, ["user"]);
    // …and on a card with a monitor on it, what the user saw beside the flag.
    const first = applyKey(state({ selected: 0 }), "d").state.gpus[0];
    assert.deepEqual(first?.evidence, ["user", "drm-connector:card1-DP-3"]);
    // The headroom default follows the flag…
    assert.equal(next.gpus[1]?.headroomGb, DISPLAY_HEADROOM_GB);
    assert.equal(next.changed, true);
  });

  it("…but a toggle never overwrites a headroom the user set themselves", () => {
    const adjusted = applyKey(state({ selected: 1 }), RIGHT).state;
    const toggled = applyKey(adjusted, "d").state;
    assert.equal(toggled.gpus[1]?.display, true);
    assert.equal(toggled.gpus[1]?.headroomGb, HEADLESS_HEADROOM_GB + HEADROOM_STEP_GB);
  });

  it("leaves the state it was given untouched", () => {
    const before = state();
    applyKey(before, "d");
    assert.equal(before.gpus[0]?.display, true);
    assert.equal(before.changed, false);
  });
});

describe("conclusion", () => {
  it("names the connected output when both signals agree", () => {
    assert.deepEqual(conclusion(gpu(), "idle-vram"), {
      text: "display attached — card1-DP-3 connected",
      token: "warning",
    });
  });

  it("falls back to the idle figure where sysfs cannot be read", () => {
    assert.match(conclusion(gpu({ connected: [] }), "idle-vram").text, /1\.21 GiB held at idle/);
  });

  it("says a flag came from the user when it did", () => {
    assert.equal(conclusion(gpu({ connected: [] }), "profile").text, "display attached — set by you");
  });

  it("flags a headless verdict that has a monitor plugged into it", () => {
    const verdict = conclusion(gpu({ display: false, evidence: [] }), "idle-vram");
    assert.equal(verdict.token, "error");
    assert.match(verdict.text, /but card1-DP-3 is connected/);
  });

  it("refuses to call a card headless before anything has been measured", () => {
    const verdict = conclusion(gpu({ display: false, idleUsedGb: undefined, connected: [], evidence: [] }), "none");
    assert.match(verdict.text, /not measured — unload to measure/);
  });

  it("does not call a card headless-by-VRAM when no VRAM reading exists", () => {
    // Observed live: the reference box with the 27B resident, where the only
    // evidence available is the connector — and it is evidence *for* display.
    const verdict = conclusion(gpu({ display: false, idleUsedGb: undefined, evidence: [] }), "none");
    assert.equal(verdict.text, "not measured — but card1-DP-3 is connected; d to set");
    assert.equal(verdict.token, "warning");
  });

  it("recommends the measured headless card", () => {
    const verdict = conclusion(gpu({ display: false, connected: [], evidence: [], idleUsedGb: 0.03 }), "idle-vram");
    assert.deepEqual(verdict, { text: "headless — preferred for models", token: "success" });
  });
});

describe("the lines it draws", () => {
  it("states the server, its version and its backend", () => {
    assert.equal(serverLine(state(), theme).trim(), "Server    127.0.0.1:8888  ✓ up  v2026.9.7  rocm");
  });

  it("says why the server line is not green when it is not", () => {
    const down = state({
      server: {
        baseUrl: "http://127.0.0.1:8888",
        up: false,
        detail: "ECONNREFUSED",
        version: undefined,
        backend: undefined,
        remote: false,
      },
    });
    assert.match(serverLine(down, theme), /✗ not answering {2}ECONNREFUSED/);
  });

  it("counts compute GPUs and the integrated ones it ignored", () => {
    assert.equal(countLine(state(), theme).trim(), "Detected 2 compute GPUs  (1 integrated, ignored)");
    assert.equal(countLine(state({ gpus: [gpu()], ignoredCount: 0 }), theme).trim(), "Detected 1 compute GPU");
  });

  it("distinguishes an idle measurement from a live one", () => {
    assert.equal(usageLine(gpu()), "1.21 GiB in use at idle");
    assert.match(usageLine(gpu({ idleUsedGb: undefined, usedGb: 21.28 })), /21\.28 GiB in use now — nothing/);
  });

  it("shows the headroom and what it leaves usable", () => {
    const lines = wizardLines(state(), 70, theme);
    assert.ok(lines.some((line) => line.includes("‹ 3.0 GiB ›") && line.includes("19.8 GiB usable")), lines.join("\n"));
    assert.equal(usable(gpu()).toFixed(2), "19.77");
  });

  it("draws the whole panel", () => {
    const text = wizardLines(state(), 70, theme).join("\n");
    // "Unsloth setup" is the frame's title now, not the body's first line.
    assert.match(text, /Server/);
    assert.match(text, /→ GPU 0/);
    assert.match(text, /▸ display attached — card1-DP-3 connected/);
    assert.match(text, /▸ headless — preferred for models/);
    assert.match(text, /d {2}toggles the display flag/);
    assert.match(text, /↑↓ move {2}←→ adjust {2}d toggle {2}e server {2}⏎ save {2}esc cancel/);
  });

  it("keeps every line inside the width it was given", () => {
    for (const width of [50, 54, 70, 120]) {
      for (const line of wizardLines(state(), width, theme)) {
        const shown = visibleWidth(line);
        assert.ok(shown <= width, `${shown} > ${width}: ${line}`);
      }
    }
  });

  it("wraps rather than clipping, so the version survives a 54-column panel", () => {
    const gated = state({
      sizing: { enabled: false, reason: "sizing needs Unsloth ≥ 2026.9 — this server reports 2026.8.4", version: "2026.8.4" },
    });
    const text = wizardLines(gated, 54, theme).join("\n");
    assert.match(text, /2026\.8\.4/);
    for (const line of wizardLines(gated, 54, theme)) assert.ok(visibleWidth(line) <= 54, line);
  });

  it("shows the sizing gate, and the reason the cross-check was skipped", () => {
    const gated = state({
      sizing: { enabled: false, reason: "sizing needs Unsloth ≥ 2026.9 — this server reports 2026.8", version: "2026.8" },
      connectorNote: "2 DRM cards for 3 GPUs — cannot line them up",
    });
    const text = wizardLines(gated, 70, theme).join("\n");
    assert.match(text, /⚠ sizing needs Unsloth ≥ 2026\.9 — this server reports 2026\.8/);
    assert.match(text, /No display cross-check — 2 DRM cards for 3 GPUs/);
  });

  it("says what is missing, and only esc, when there is no topology", () => {
    const text = wizardLines(state({ gpus: [], degradedReason: "the server did not answer /api/system" }), 70, theme).join("\n");
    assert.match(text, /⚠ No GPU topology — the server did not answer/);
    assert.match(text, /Models still load and unload; sizing is disabled/);
    assert.match(text, /esc close/);
    assert.doesNotMatch(text, /⏎ save/);
  });
});

describe("wizardSummary", () => {
  it("is one honest line for a terminal that cannot draw the panel", () => {
    assert.equal(wizardSummary(state()), "⬢ unsloth — 2 compute GPUs · display: GPU 0");
  });

  it("does not claim a display GPU it has not found", () => {
    const headless = state({ gpus: state().gpus.map((entry) => ({ ...entry, display: false })) });
    assert.match(wizardSummary(headless), /no display GPU detected/);
  });

  it("carries the degraded reason rather than reporting a working machine", () => {
    assert.equal(wizardSummary(state({ gpus: [], degradedReason: "no GPUs" })), "◌ unsloth — no GPUs");
  });
});

describe("profileFrom", () => {
  const now = new Date("2026-09-21T12:00:00.000Z");

  it("writes what the user accepted, with the evidence behind it", () => {
    const profile = profileFrom(state(), undefined, now);
    assert.equal(profile.version, PROFILE_VERSION);
    assert.equal(profile.detectedAt, "2026-09-21T12:00:00.000Z");
    assert.equal(profile.unslothVersion, "2026.9.7");
    assert.equal(profile.backend, "rocm");
    assert.deepEqual(profile.gpus[0], {
      index: 0,
      display: true,
      headroomGiB: DISPLAY_HEADROOM_GB,
      displayEvidence: ["idle-vram", "drm-connector:card1-DP-3"],
      name: "AMD Radeon Graphics",
      totalGiB: 23.98,
      idleUsedGiB: 1.21,
    });
  });

  it("records no evidence for a card it is not calling a display", () => {
    assert.deepEqual(profileFrom(state(), undefined, now).gpus[1]?.displayEvidence, []);
  });

  it("carries over the policy and calibration it did not collect", () => {
    const existing: MachineProfile = {
      version: PROFILE_VERSION,
      gpus: [],
      policy: { preferHeadless: true, kvDtype: "q8_0" },
      calibration: { estimateDeltaGiB: 0.4, samples: 2 },
    };
    const profile = profileFrom(state(), existing, now);
    assert.deepEqual(profile.policy, { preferHeadless: true, kvDtype: "q8_0" });
    assert.deepEqual(profile.calibration, { estimateDeltaGiB: 0.4, samples: 2 });
  });

  it("leaves a GPU's hand-added keys alone while replacing what it owns", () => {
    const existing: MachineProfile = {
      version: PROFILE_VERSION,
      gpus: [{ index: 0, display: false, headroomGiB: 9, displayEvidence: [], note: "mine" }],
    };
    const written = profileFrom(state(), existing, now).gpus[0];
    assert.equal(written?.["note"], "mine");
    assert.equal(written?.display, true);
    assert.equal(written?.headroomGiB, DISPLAY_HEADROOM_GB);
  });

  it("omits an idle figure it never measured instead of writing a zero", () => {
    const unmeasured = state({ gpus: [gpu({ idleUsedGb: undefined })] });
    assert.equal("idleUsedGiB" in (profileFrom(unmeasured, undefined, now).gpus[0] ?? {}), false);
  });
});

/* --------------------------------------------------------------------------
 * The address field and the remote screen
 * ------------------------------------------------------------------------ */

const REMOTE = {
  baseUrl: "http://192.168.1.40:8888",
  up: true,
  detail: undefined,
  version: "2026.9.7",
  backend: "rocm",
  remote: true,
};

const BACKSPACE = "\u007f";

/** Wrapped, indented lines as one sentence — the text, not its line breaks. */
function flatten(lines: readonly string[]): string {
  return lines.join(" ").replace(/\s+/g, " ").trim();
}

/** Feed a string through the reducer one key at a time, as a terminal would. */
function type(start: WizardState, keys: string[]): WizardState {
  return keys.reduce((current, key) => applyKey(current, key).state, start);
}

describe("the address field", () => {
  it("opens on the address already configured, so a port is two keystrokes", () => {
    const opened = applyKey(state(), "e").state;
    assert.equal(opened.editing, "127.0.0.1:8888");
  });

  it("takes text and backspace", () => {
    const edited = type(state(), ["e", BACKSPACE, BACKSPACE, BACKSPACE, BACKSPACE, "9", "0", "0", "0"]);
    assert.equal(edited.editing, "127.0.0.1:9000");
  });

  it("swallows the keys that mean something behind it", () => {
    // `d` in a hostname must not toggle a display flag, and an arrow must not
    // move the selection under a half-typed address.
    const edited = type(state(), ["e", "d", DOWN, LEFT]);
    assert.equal(edited.editing, "127.0.0.1:8888d");
    assert.equal(edited.gpus[0]?.display, true);
    assert.equal(edited.selected, 0);
  });

  it("ignores an escape sequence rather than typing it into the hostname", () => {
    const edited = type(state(), ["e", UP]);
    assert.equal(edited.editing, "127.0.0.1:8888");
  });

  it("escape abandons the edit, not the wizard, and changes no address", () => {
    const { state: after, outcome } = applyKey(type(state(), ["e", "9"]), ESC);
    assert.equal(outcome, "open");
    assert.equal(after.editing, undefined);
    assert.equal(after.server.baseUrl, "http://127.0.0.1:8888");
  });

  it("asks the caller to test the address on ⏎, rather than saving anything", () => {
    const { outcome } = applyKey(type(state(), ["e", "9"]), ENTER);
    assert.equal(outcome, "endpoint");
  });

  it("draws the field in place of the status, with what each key does", () => {
    const editing = type(state(), ["e"]);
    assert.match(serverLine(editing, theme), /\[ 127\.0\.0\.1:8888_ \]/);
    assert.match(endpointLines(editing, 70, theme).join("\n"), /⏎ test and use {3}esc keep 127\.0\.0\.1:8888/);
  });

  it("shows why an address was refused, with the field still open", () => {
    const refused = { ...type(state(), ["e"]), editError: "nonsense is not an address" };
    assert.match(endpointLines(refused, 70, theme).join("\n"), /✗ nonsense is not an address/);
  });

  it("hides the GPUs while the field is open", () => {
    // They belong to the address being replaced, and `⏎` must not read as
    // "save those".
    const drawn = wizardLines(type(state(), ["e"]), 70, theme).join("\n");
    assert.doesNotMatch(drawn, /GPU 0/);
    assert.match(drawn, /⏎ test and use/);
  });
});

describe("the remote screen", () => {
  it("marks the endpoint, and says which three things are off", () => {
    const remote = state({ server: REMOTE });
    assert.match(serverLine(remote, theme), /192\.168\.1\.40:8888.*⌁ remote/);
    // Ahead of the status, so a long health detail cannot clip the badge off
    // the end of the line.
    const failing = state({
      server: { ...REMOTE, up: false, detail: "/api/health: getaddrinfo ENOTFOUND gpubox.invalid" },
    });
    assert.ok(serverLine(failing, theme).indexOf("⌁ remote") < serverLine(failing, theme).indexOf("not answering"));
    const lines = flatten(endpointLines(remote, 70, theme));
    assert.match(lines, /Autostart, freeing VRAM and the display cross-check/);
  });

  it("says nothing of the sort about a server on this machine", () => {
    assert.doesNotMatch(serverLine(state(), theme), /remote/);
    assert.deepEqual(endpointLines(state(), 70, theme), []);
  });

  it("names the bind address when a remote endpoint does not answer", () => {
    // Nobody guesses `-H 0.0.0.0` unaided, and the fix is on another machine.
    const down = state({
      server: { ...REMOTE, up: false, detail: "ECONNREFUSED" },
      degradedReason: "the server did not answer /api/system",
    });
    const lines = flatten(endpointLines(down, 70, theme));
    assert.match(lines, /bound to loopback on its own machine \(-H 0\.0\.0\.0\)/);
    // One key hint per screen: the degraded footer is what offers `e`.
    assert.doesNotMatch(lines, /e edit again/);
    assert.match(wizardLines(down, 70, theme).join("\n"), /e server {3}esc close/);
  });

  it("offers the address field on a degraded screen, where it is the only fix", () => {
    const degraded = state({ server: REMOTE, degradedReason: "the server did not answer /api/system" });
    assert.match(wizardLines(degraded, 70, theme).join("\n"), /e server {3}esc close/);
  });

  it("names the host in the one-line summary, where there is no badge to draw", () => {
    assert.match(wizardSummary(state({ server: REMOTE })), /at 192\.168\.1\.40:8888/);
    assert.doesNotMatch(wizardSummary(state()), /at 127/);
  });
});

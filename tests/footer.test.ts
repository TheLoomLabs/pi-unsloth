import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  METER_COLUMNS,
  NARROW_COLUMNS,
  formatVram,
  renderFooter,
  vramFraction,
  type FooterSnapshot,
  type FooterTheme,
} from "../src/ui/footer.ts";

/**
 * A theme that records tokens instead of emitting colour, so a test can assert
 * "this state is drawn as a warning" without parsing ANSI.
 *
 * `token` is the **first** one now: the line is painted a chunk at a time, so
 * the last token belongs to a separator rather than to the state.
 */
function probe(): FooterTheme & { token: string | undefined; tokens: string[] } {
  const tokens: string[] = [];
  const recorder = {
    tokens,
    get token(): string | undefined {
      return tokens[0];
    },
    fg(color: string, text: string) {
      if (text !== "") tokens.push(color);
      return text;
    },
  };
  return recorder as FooterTheme & { token: string | undefined; tokens: string[] };
}

/** Wide enough for every chunk, but not for the meter. */
const WIDE = METER_COLUMNS - 1;

/** Two identical cards, one of them busy — the example. */
const TWO_GPUS = [
  { index: 0, usedGb: 21.4, totalGb: 24 },
  { index: 1, usedGb: 18.2, totalGb: 24 },
];

function snap(overrides: Partial<FooterSnapshot> = {}): FooterSnapshot {
  return {
    server: "up",
    activeModelLabel: undefined,
    activeContextWindow: undefined,
    loading: undefined,
    streaming: false,
    gpus: [],
    ...overrides,
  };
}

describe("formatVram", () => {
  it("totals the machine rather than itemising it — the panel has the bars", () => {
    assert.equal(formatVram(TWO_GPUS), "39.6/48.0 GiB");
  });

  it("reads the same on a single GPU", () => {
    assert.equal(formatVram([{ index: 1, usedGb: 0, totalGb: 24 }]), "0.0/24.0 GiB");
  });

  it("says nothing when no GPU has been seen", () => {
    assert.equal(formatVram([]), undefined);
  });
});

describe("vramFraction", () => {
  it("is how full the machine is, across its compute GPUs", () => {
    assert.equal((vramFraction(TWO_GPUS) ?? 0).toFixed(3), "0.825");
  });

  it("has no opinion without cards, or without sizes", () => {
    assert.equal(vramFraction([]), undefined);
    assert.equal(vramFraction([{ index: 0, usedGb: 1, totalGb: 0 }]), undefined);
  });
});

describe("renderFooter — the six states", () => {
  it("loaded and idle", () => {
    const theme = probe();
    const line = renderFooter(
      snap({ activeModelLabel: "a-model", activeContextWindow: 196608, gpus: TWO_GPUS }),
      WIDE,
      theme,
    );
    assert.equal(line, "⬢ a-model · 192K · 39.6/48.0 GiB");
    assert.equal(theme.token, "accent");
  });

  it("generating — same line plus the marker", () => {
    const theme = probe();
    const line = renderFooter(
      snap({ activeModelLabel: "a-model", activeContextWindow: 196608, gpus: TWO_GPUS, streaming: true }),
      WIDE,
      theme,
    );
    assert.equal(line, "⬢ a-model · 192K · 39.6/48.0 GiB ▸");
    assert.equal(theme.token, "success");
  });

  it("loading, with a percentage once the server reports one", () => {
    const theme = probe();
    const line = renderFooter(snap({ loading: { label: "a-model", fraction: 0.68 } }), WIDE, theme);
    assert.equal(line, "◐ loading a-model · 68%");
    assert.equal(theme.token, "warning");
  });

  it("loading, with no percentage before the server knows the total", () => {
    const line = renderFooter(snap({ loading: { label: "a-model", fraction: undefined } }), WIDE, probe());
    assert.equal(line, "◐ loading a-model");
  });

  it("server up, nothing loaded", () => {
    const theme = probe();
    const line = renderFooter(snap({ gpus: [{ index: 0, usedGb: 0, totalGb: 24 }] }), WIDE, theme);
    assert.equal(line, "○ unsloth idle · 0.0/24.0 GiB");
    assert.equal(theme.token, "muted");
  });

  it("server down is dim, not an error", () => {
    const theme = probe();
    assert.equal(renderFooter(snap({ server: "offline" }), WIDE, theme), "◌ unsloth offline");
    assert.equal(theme.token, "dim");
  });

  it("a server that would not come up is an error", () => {
    const theme = probe();
    assert.equal(renderFooter(snap({ server: "unreachable" }), WIDE, theme), "⚠ unsloth unreachable");
    assert.equal(theme.token, "error");
  });
});

describe("renderFooter — the other states", () => {
  it("names a rejected key rather than calling it unreachable", () => {
    const theme = probe();
    assert.equal(renderFooter(snap({ server: "unauthorized" }), WIDE, theme), "⚠ unsloth unauthorized");
    assert.equal(theme.token, "error");
  });

  it("shares the loading shape while the server is starting", () => {
    const theme = probe();
    assert.equal(renderFooter(snap({ server: "starting" }), WIDE, theme), "◐ starting unsloth");
    assert.equal(theme.token, "warning");
  });

  it("says nothing at all before anything has been observed", () => {
    assert.equal(renderFooter(snap({ server: "unknown" }), WIDE, probe()), undefined);
  });

  it("clamps a fraction the server reports out of range", () => {
    assert.equal(renderFooter(snap({ loading: { label: "m", fraction: 1.4 } }), WIDE, probe()), "◐ loading m · 100%");
    assert.equal(renderFooter(snap({ loading: { label: "m", fraction: -1 } }), WIDE, probe()), "◐ loading m · 0%");
  });
});

describe("renderFooter — narrow terminals", () => {
  const full = snap({ activeModelLabel: "a-model", activeContextWindow: 196608, gpus: TWO_GPUS });

  it("keeps only the model name below 60 columns", () => {
    assert.equal(renderFooter(full, NARROW_COLUMNS - 1, probe()), "⬢ a-model");
  });

  it("keeps the whole line at exactly 60 columns", () => {
    assert.equal(renderFooter(full, NARROW_COLUMNS, probe()), "⬢ a-model · 192K · 39.6/48.0 GiB");
  });

  it("keeps the streaming marker, which is state rather than detail", () => {
    const line = renderFooter({ ...full, streaming: true }, NARROW_COLUMNS - 1, probe());
    assert.equal(line, "⬢ a-model ▸");
  });
});

describe("renderFooter — the meter", () => {
  const full = snap({ activeModelLabel: "a-model", activeContextWindow: 196608, gpus: TWO_GPUS });

  it("draws a meter before the figure once there is room for one", () => {
    const line = renderFooter(full, METER_COLUMNS, probe()) ?? "";
    assert.match(line, /▓+░* 39\.6\/48\.0 GiB$/);
  });

  it("drops the bar and keeps the figure when there is not", () => {
    const line = renderFooter(full, METER_COLUMNS - 1, probe()) ?? "";
    assert.doesNotMatch(line, /▓/);
    assert.match(line, /39\.6\/48\.0 GiB$/);
  });

  it("colours the meter by how full the machine is", () => {
    const nearlyFull = snap({
      activeModelLabel: "a-model",
      gpus: [{ index: 0, usedGb: 23.5, totalGb: 24 }],
    });
    const theme = probe();
    renderFooter(nearlyFull, METER_COLUMNS, theme);
    assert.ok(theme.tokens.includes("error"), theme.tokens.join(","));
  });
});

describe("renderFooter — the model's name", () => {
  it("does not repeat a context the name already carries", () => {
    const line = renderFooter(
      snap({ activeModelLabel: "a-model-192K", activeContextWindow: 196608, gpus: [] }),
      WIDE,
      probe(),
    );
    assert.equal(line, "⬢ a-model-192K");
  });
});

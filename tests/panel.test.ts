import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { CatalogueEntry, LocalModel } from "../src/api/models.ts";
import type { Overrides } from "../src/api/lifecycle.ts";
import {
  buildRows,
  emptyLines,
  hintLines,
  panelSummary,
  placementOf,
  renderGpuBars,
  renderRow,
  rowTier,
  statusCell,
  visibleRows,
  type PanelData,
  type PanelRow,
  type PanelTheme,
} from "../src/ui/panel.ts";
import { visibleWidth } from "@earendil-works/pi-tui";

/**
 * A theme that tags rather than colours, so a test can assert "this is drawn as
 * an error" without parsing ANSI — and so every assertion about *width* is an
 * assertion about the real string, not about escape codes.
 */
function probe(): PanelTheme & { tokens: string[] } {
  const seen: string[] = [];
  return {
    tokens: seen,
    fg(color: string, text: string) {
      if (text !== "") seen.push(color);
      return text;
    },
    bold(text: string) {
      return text;
    },
  } as PanelTheme & { tokens: string[] };
}

/** What the reference box reports, trimmed to the fields rows are built from. */
const ENTRIES: CatalogueEntry[] = [
  {
    id: "Qwen3.8-27B",
    displayName: undefined,
    quant: "Q8_0",
    loaded: true,
    contextLength: 196608,
    maxContextLength: 262144,
    nativeContextLength: 262144,
  },
  {
    id: "ggml-org/Qwen3-4B-GGUF",
    displayName: "Qwen3-4B-GGUF",
    quant: "Q4_K_M",
    loaded: false,
    contextLength: undefined,
    maxContextLength: undefined,
    nativeContextLength: undefined,
  },
];

const LOCAL: LocalModel[] = [
  {
    id: "Qwen3.8-27B",
    displayName: "Qwen3.8-27B",
    path: "/models/Qwen3.8-27B",
    source: "models_dir",
    repoId: undefined,
    modelFormat: "gguf",
    task: "text",
    partial: false,
  },
  {
    id: "ggml-org/Qwen3-4B-GGUF",
    displayName: "Qwen3-4B-GGUF",
    path: "/cache/Qwen3-4B",
    source: "hf_cache",
    repoId: "ggml-org/Qwen3-4B-GGUF",
    modelFormat: "gguf",
    task: "text",
    partial: false,
  },
];

const OVERRIDES: Overrides = {
  "/models/Qwen3.8-27B:Q8_0": {
    custom_context_length: 196608,
    kv_cache_dtype: "q8_0",
    speculative_type: "mtp",
    tensor_parallel: true,
    gpu_ids: [0, 1],
  },
  "ggml-org/Qwen3-4B-GGUF:Q4_K_M": { custom_context_length: 40960, gpu_ids: [1] },
};

const TWO_GPUS = [
  { index: 0, usedGb: 21.4, totalGb: 24 },
  { index: 1, usedGb: 18.2, totalGb: 24 },
];

function rows(overrides: Overrides = OVERRIDES, gpuIndices: number[] = [0, 1]): PanelRow[] {
  return buildRows({ entries: ENTRIES, local: LOCAL, overrides, gpuIndices });
}

describe("buildRows", () => {
  it("reads each model's own tuned context and placement", () => {
    const [big, small] = rows();
    assert.equal(big?.name, "Qwen3.8-27B");
    assert.equal(big?.contextTokens, 196608);
    assert.equal(big?.placement, "tp 0+1");
    assert.equal(big?.fit, "sized");
    assert.equal(small?.contextTokens, 40960);
    assert.equal(small?.placement, "gpu 1");
  });

  it("puts the loaded model first — it is the one holding the VRAM", () => {
    assert.deepEqual(rows().map((row) => row.loaded), [true, false]);
  });

  it("resolves the path a load needs, hub id or absolute path", () => {
    assert.deepEqual(rows().map((row) => row.modelPath), [
      "/models/Qwen3.8-27B",
      "ggml-org/Qwen3-4B-GGUF",
    ]);
  });

  it("calls a model with no override `not sized`, not `won't fit`", () => {
    const [big] = rows({});
    assert.equal(big?.fit, "unsized");
    assert.equal(big?.contextTokens, undefined);
    assert.deepEqual(statusCell(big as PanelRow), { text: "not sized", token: "dim" });
  });

  it("says `won't fit` when the tuned GPUs are not on this machine", () => {
    const [big] = rows(OVERRIDES, [0]);
    assert.equal(big?.fit, "wont-fit");
    assert.deepEqual(statusCell(big as PanelRow), { text: "won't fit", token: "error" });
  });

  it("flags a half-downloaded model rather than offering to load it", () => {
    const partial = LOCAL.map((model) => ({ ...model, partial: true }));
    const [big] = buildRows({ entries: ENTRIES, local: partial, overrides: OVERRIDES, gpuIndices: [0, 1] });
    assert.equal(big?.fit, "incomplete");
  });

  it("flags a catalogue entry with nothing behind it on disk", () => {
    const [big] = buildRows({ entries: ENTRIES, local: [], overrides: OVERRIDES, gpuIndices: [0, 1] });
    assert.equal(big?.fit, "missing");
    assert.equal(big?.modelPath, undefined);
  });
});

describe("placementOf", () => {
  it("names tensor-parallel and single-GPU placements differently", () => {
    assert.deepEqual(placementOf({ tensor_parallel: true, gpu_ids: [0, 1] }, [0, 1]), {
      text: "tp 0+1",
      fits: true,
    });
    assert.deepEqual(placementOf({ gpu_ids: [1] }, [0, 1]), { text: "gpu 1", fits: true });
  });

  it("claims nothing about fit when the topology is unknown", () => {
    assert.equal(placementOf({ gpu_ids: [7] }, []).fits, true);
  });

  it("says nothing at all when the override names no placement", () => {
    assert.deepEqual(placementOf({ custom_context_length: 4096 }, [0]), { text: undefined, fits: true });
  });
});

describe("renderRow", () => {
  const theme = probe();

  it("draws all four columns wide", () => {
    const [big] = rows();
    const line = renderRow(big as PanelRow, 50, theme, false);
    assert.match(line, /●\s+Qwen3\.8-27B\s+Q8_0\s+192K\s+tp 0\+1/);
    assert.equal(visibleWidth(line), 50);
  });

  it("marks a cold model with ○", () => {
    const [, small] = rows();
    assert.ok(renderRow(small as PanelRow, 50, theme, false).startsWith("○ "));
  });

  it("keeps the state word when the columns have to go", () => {
    const [big] = rows({});
    const line = renderRow(big as PanelRow, 30, theme, false);
    assert.equal(rowTier(30), "compact");
    assert.match(line, /not sized/);
    assert.doesNotMatch(line, /Q8_0/);
    assert.ok(visibleWidth(line) <= 30);
  });

  it("gives a sized model's context the column the state word would have used", () => {
    const [big] = rows();
    assert.match(renderRow(big as PanelRow, 30, theme, false), /192K/);
  });

  it("never overflows, at any width it is asked to draw at", () => {
    const [big] = rows();
    for (let width = 8; width <= 80; width++) {
      assert.ok(
        visibleWidth(renderRow(big as PanelRow, width, theme, false)) <= width,
        `row overflowed at width ${width}`,
      );
    }
  });
});

describe("renderGpuBars", () => {
  it("prints the figure beside the bar, and tags the display GPU", () => {
    const theme = probe();
    const lines = renderGpuBars(TWO_GPUS, [0], 52, theme);
    assert.equal(lines.length, 2);
    assert.match(lines[0] ?? "", /GPU 0.*21\.4\/24\.0\s+display/);
    assert.match(lines[1] ?? "", /GPU 1.*18\.2\/24\.0/);
    assert.doesNotMatch(lines[1] ?? "", /display/);
  });

  it("drops the display tag before it drops the figure beside it", () => {
    const line = renderGpuBars(TWO_GPUS, [0], 28, probe())[0] ?? "";
    assert.match(line, /21\.4\/24\.0/);
    assert.doesNotMatch(line, /display/);
  });

  it("colours a bar by how full it is", () => {
    const tokenFor = (usedGb: number): string => {
      const theme = probe();
      renderGpuBars([{ index: 0, usedGb, totalGb: 24 }], [], 52, theme);
      return theme.tokens.find((token) => ["accent", "warning", "error"].includes(token)) ?? "none";
    };
    assert.equal(tokenFor(12), "accent");
    assert.equal(tokenFor(21), "warning");
    assert.equal(tokenFor(23.5), "error");
  });

  it("says nothing when no GPU has been seen", () => {
    assert.deepEqual(renderGpuBars([], undefined, 52, probe()), []);
  });

  it("drops the bar rather than the figure when the panel is narrow", () => {
    const line = renderGpuBars(TWO_GPUS, undefined, 24, probe())[0] ?? "";
    assert.match(line, /21\.4\/24\.0/);
    assert.doesNotMatch(line, /[▓░]/);
    assert.ok(visibleWidth(line) <= 24);
  });

  it("never overflows, at any width it is asked to draw at", () => {
    for (let width = 12; width <= 100; width++) {
      for (const line of renderGpuBars(TWO_GPUS, [0], width, probe())) {
        assert.ok(visibleWidth(line) <= width, `bar overflowed at width ${width}`);
      }
    }
  });
});

describe("the lines around the list", () => {
  const theme = probe();

  function data(overrides: Partial<PanelData> = {}): PanelData {
    return { rows: rows(), gpus: TWO_GPUS, displayGpus: [0], problem: undefined, ...overrides };
  }

  it("shortens the key hints rather than overflowing them", () => {
    const wide = hintLines(true, 60, theme)[0] ?? "";
    const narrow = hintLines(true, 30, theme)[0] ?? "";
    assert.match(wide, /r refresh/);
    assert.doesNotMatch(narrow, /r refresh/);
    assert.ok(visibleWidth(narrow) <= 30);
  });

  it("keeps ⏎, a and esc at every width — they are the row states' answers", () => {
    for (const width of [60, 54, 40, 30, 24]) {
      const hint = hintLines(true, width, theme)[0] ?? "";
      assert.match(hint, /⏎ load/, `width ${width}`);
      assert.match(hint, /a size/, `width ${width}`);
      assert.match(hint, /esc/, `width ${width}`);
    }
  });

  it("explains an empty list rather than showing an empty box", () => {
    assert.match(emptyLines(data({ rows: [] }), 52, theme)[0] ?? "", /no models downloaded/);
    assert.match(
      emptyLines(data({ rows: [], problem: "ECONNREFUSED" }), 52, theme)[0] ?? "",
      /unavailable — ECONNREFUSED/,
    );
  });

  it("summarises itself in one line for a terminal too narrow to draw in", () => {
    assert.equal(panelSummary(data()), "⬢ Qwen3.8-27B · 2 models · 21.4+18.2/48.0 GiB");
    assert.match(panelSummary(data({ rows: [] })), /^○ unsloth idle/);
    assert.match(panelSummary(data({ problem: "down" })), /^◌ unsloth — down/);
  });
});

describe("visibleRows", () => {
  it("shows every model when the terminal has room", () => {
    assert.equal(visibleRows(4, 44), 4);
  });

  it("caps the list so a long catalogue does not become the whole panel", () => {
    assert.equal(visibleRows(40, 60), 12);
  });

  it("scrolls rather than pushing the key hints off the bottom", () => {
    assert.ok(visibleRows(40, 24) < 12);
  });

  it("still shows something in a terminal with no room at all", () => {
    assert.equal(visibleRows(40, 8), 3);
  });
});

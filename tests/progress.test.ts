import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { visibleWidth } from "@earendil-works/pi-tui";

import { ERROR_FRACTION, WARNING_FRACTION, meterBar, meterToken } from "../src/ui/draw.ts";
import {
  bytesLine,
  describeConfig,
  phaseTitle,
  renderProgress,
  type ProgressSnapshot,
  type ProgressTheme,
} from "../src/ui/progress.ts";

function probe(): ProgressTheme & { tokens: string[] } {
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
  } as ProgressTheme & { tokens: string[] };
}

const GIB = 1024 ** 3;

describe("meterBar", () => {
  const theme = probe();

  it("fills exactly the width it is given", () => {
    for (const fraction of [0, 0.01, 0.5, 0.999, 1, 2, Number.NaN]) {
      assert.equal(visibleWidth(meterBar(fraction, 20, theme)), 20, `fraction ${fraction}`);
    }
  });

  it("shows a sliver rather than nothing when a little is in use", () => {
    assert.match(meterBar(0.01, 20, probe()), /^▓░+$/);
  });

  it("is empty only when nothing is in use", () => {
    assert.match(meterBar(0, 8, probe()), /^░{8}$/);
  });

  it("turns warning then error at the documented thresholds", () => {
    assert.equal(meterToken(WARNING_FRACTION - 0.01), "accent");
    assert.equal(meterToken(WARNING_FRACTION), "warning");
    assert.equal(meterToken(ERROR_FRACTION), "error");
  });
});

describe("phaseTitle", () => {
  it("heads the panel with the server's phase", () => {
    assert.equal(phaseTitle("resolving", "Qwen3.8-27B"), "Resolving Qwen3.8-27B");
    assert.equal(phaseTitle("warming up", "Qwen3.8-27B"), "Warming up Qwen3.8-27B");
  });

  it("does not print a phase it does not recognise", () => {
    assert.equal(phaseTitle("quantising-the-flux", "X"), "Loading X");
    assert.equal(phaseTitle(undefined, "X"), "Loading X");
  });
});

describe("bytesLine", () => {
  it("reads the reference 27B load the way draws it", () => {
    assert.equal(bytesLine(19.8 * GIB, 29 * GIB), "19.8 / 29.0 GiB");
  });

  it("shows what it has when the server has not sized the job", () => {
    assert.equal(bytesLine(2 * GIB, undefined), "2.0 GiB");
    assert.equal(bytesLine(undefined, undefined), undefined);
  });
});

describe("describeConfig", () => {
  it("states the tuned config in the user's terms", () => {
    assert.equal(
      describeConfig({
        custom_context_length: 196608,
        kv_cache_dtype: "q8_0",
        speculative_type: "mtp",
        tensor_parallel: true,
        gpu_ids: [0, 1],
      }),
      "tensor-parallel 0+1 · ctx 192K · kv q8_0 · mtp",
    );
  });

  it("names a single GPU as a single GPU", () => {
    assert.equal(describeConfig({ custom_context_length: 40960, gpu_ids: [1] }), "gpu 1 · ctx 40K");
  });

  it("invents nothing for a model with no override", () => {
    assert.equal(describeConfig(undefined), undefined);
    assert.equal(describeConfig({}), undefined);
  });
});

describe("renderProgress", () => {
  const theme = probe();

  function snap(overrides: Partial<ProgressSnapshot> = {}): ProgressSnapshot {
    return {
      label: "Qwen3.8-27B",
      loading: { label: "Qwen3.8-27B", fraction: 0.68, phase: "loading", loadedBytes: 19.8 * GIB, totalBytes: 29 * GIB },
      config: "tensor-parallel 0+1 · ctx 192K · kv q8_0 · mtp",
      error: undefined,
      ...overrides,
    };
  }

  it("draws the bar, the percentage, the bytes and the config", () => {
    const lines = renderProgress(snap(), 52, theme);
    const text = lines.join("\n");
    // The phase heads the frame now (`phaseTitle`), not the body.
    assert.equal(phaseTitle("loading", "Qwen3.8-27B"), "Loading Qwen3.8-27B");
    assert.match(text, /68%/);
    assert.match(text, /19\.8 \/ 29\.0 GiB/);
    assert.match(text, /tensor-parallel 0\+1/);
    assert.match(text, /esc  continue in background/);
    for (const line of lines) assert.ok(visibleWidth(line) <= 52);
  });

  it("does not invent a percentage before the server reports one", () => {
    const text = renderProgress(
      snap({ label: "X", loading: { label: "X", fraction: undefined, phase: "resolving" } }),
      52,
      theme,
    ).join("\n");
    assert.equal(phaseTitle("resolving", "X"), "Resolving X");
    assert.match(text, /working…/);
    assert.doesNotMatch(text, /0%/);
  });

  it("keeps the server's own reason on screen until it is dismissed", () => {
    const text = renderProgress(snap({ error: "not enough VRAM on GPU 1", loading: undefined }), 52, theme).join("\n");
    assert.match(text, /✗ not enough VRAM on GPU 1/);
    assert.match(text, /esc  dismiss/);
    assert.doesNotMatch(text, /continue in background/);
  });

  it("never overflows, at any width it is asked to draw at", () => {
    for (let width = 20; width <= 100; width++) {
      for (const line of renderProgress(snap(), width, probe())) {
        assert.ok(visibleWidth(line) <= width, `progress overflowed at width ${width}`);
      }
    }
  });
});

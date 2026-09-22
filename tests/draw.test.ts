import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { visibleWidth } from "@earendil-works/pi-tui";

import {
  FRAME_CHROME,
  MAX_OVERLAY_COLUMNS,
  MIN_OVERLAY_COLUMNS,
  frameLines,
  overlayWidth,
  type FrameTheme,
} from "../src/ui/draw.ts";

/** A theme that tags rather than colours, so widths are the real thing. */
function probe(): FrameTheme & { tokens: string[] } {
  const tokens: string[] = [];
  return {
    tokens,
    fg(color: string, text: string) {
      if (text !== "") tokens.push(color);
      return text;
    },
    bold(text: string) {
      return text;
    },
  } as FrameTheme & { tokens: string[] };
}

const theme = probe();

describe("overlayWidth", () => {
  it("is a card, not a column of the screen", () => {
    // The ultrawide case that made this necessary: a 250-column terminal was
    // giving a 130-column panel.
    assert.equal(overlayWidth(250), MAX_OVERLAY_COLUMNS);
    assert.equal(overlayWidth(120), MAX_OVERLAY_COLUMNS);
  });

  it("takes what it can get on a narrow terminal, less a margin", () => {
    assert.equal(overlayWidth(60), 56);
    assert.equal(overlayWidth(54), 50);
  });

  it("never goes below the floor it would refuse to draw at", () => {
    assert.equal(overlayWidth(40), MIN_OVERLAY_COLUMNS);
  });
});

describe("frameLines", () => {
  it("draws a rounded box with the title on its top edge", () => {
    const lines = frameLines("Unsloth", ["  hello"], 30, theme);
    assert.equal(lines.length, 3);
    assert.match(lines[0] ?? "", /^╭─ Unsloth ─+╮$/);
    assert.match(lines[1] ?? "", /^│ {3}hello +│$/);
    assert.match(lines[2] ?? "", /^╰─+╯$/);
  });

  it("makes every line exactly as wide as the box", () => {
    const body = ["", "  short", "  a much longer line than the others"];
    for (const width of [50, 62, MAX_OVERLAY_COLUMNS]) {
      for (const line of frameLines("Title", body, width, theme)) {
        assert.equal(visibleWidth(line), width, `${width}: ${line}`);
      }
    }
  });

  it("clips a line that would push the right border out of true", () => {
    const [, line] = frameLines("T", ["x".repeat(200)], 40, theme);
    assert.equal(visibleWidth(line ?? ""), 40);
  });

  it("keeps the box when the title would not fit on its edge", () => {
    const lines = frameLines("a title far longer than this narrow box", ["  x"], 20, theme);
    assert.equal(visibleWidth(lines[0] ?? ""), 20);
    assert.match(lines[0] ?? "", /^╭─+╮$/);
  });

  it("gives the body four columns less than the box, for its borders and gutters", () => {
    const width = 60;
    const body = ["  " + "y".repeat(width - FRAME_CHROME - 2)];
    const [, line] = frameLines("T", body, width, theme);
    assert.equal(visibleWidth(line ?? ""), width);
    assert.match(line ?? "", /y │$/);
  });

  it("draws the edges dim and the title in the accent", () => {
    const tagged = probe();
    frameLines("Unsloth", ["  body"], 30, tagged);
    assert.ok(tagged.tokens.includes("dim"));
    assert.ok(tagged.tokens.includes("accent"));
  });
});

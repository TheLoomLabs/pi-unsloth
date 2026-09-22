import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { visibleWidth } from "@earendil-works/pi-tui";

import {
  LOGO_FULL_COLUMNS,
  LOGO_FULL_ROWS,
  LOGO_SHORT_COLUMNS,
  LOGO_SHORT_ROWS,
  logoLines,
  type LogoTheme,
} from "../src/ui/logo.ts";

/** A theme that tags rather than colours, so the widths are the real thing. */
function probe(): LogoTheme & { tokens: string[] } {
  const tokens: string[] = [];
  return {
    tokens,
    fg(color: string, text: string) {
      if (text.trim() !== "") tokens.push(color);
      return text;
    },
  } as LogoTheme & { tokens: string[] };
}

const theme = probe();

describe("logoLines", () => {
  it("draws the sloth on its branch, with the name beside it", () => {
    const lines = logoLines(70, 20, theme);
    assert.equal(lines.length, LOGO_FULL_ROWS);
    const text = lines.join("\n");
    assert.match(text, /\(o {2}o\)/);
    assert.match(text, /█ █ █▄█ █▀▀ █ {3}█▀█ ▀█▀ █ █/);
  });

  it("hangs both halves from one branch", () => {
    const branch = logoLines(70, 20, theme)[0] ?? "";
    // Long enough to reach past the wordmark, or the name floats unsupported.
    assert.ok(visibleWidth(branch) >= LOGO_FULL_COLUMNS - 1, branch);
    assert.match(branch, /^ *─+$/);
  });

  it("sheds the sloth when the overlay is too narrow for both", () => {
    const lines = logoLines(LOGO_FULL_COLUMNS - 1, 20, theme);
    assert.equal(lines.length, LOGO_SHORT_ROWS);
    assert.doesNotMatch(lines.join("\n"), /\(o {2}o\)/);
  });

  it("sheds the sloth when the screen cannot spare the rows", () => {
    assert.equal(logoLines(70, LOGO_FULL_ROWS - 1, theme).length, LOGO_SHORT_ROWS);
  });

  it("draws nothing at all rather than half a wordmark", () => {
    assert.deepEqual(logoLines(70, LOGO_SHORT_ROWS - 1, theme), []);
    assert.deepEqual(logoLines(LOGO_SHORT_COLUMNS - 1, 20, theme), []);
    assert.deepEqual(logoLines(0, 0, theme), []);
  });

  it("stays inside the width it was given, in both forms", () => {
    for (const width of [LOGO_SHORT_COLUMNS, 50, LOGO_FULL_COLUMNS, 70]) {
      for (const line of logoLines(width, 20, theme)) {
        const shown = visibleWidth(line);
        assert.ok(shown <= width, `${shown} > ${width}: ${line}`);
      }
    }
  });

  it("is built from single-column characters, so it cannot tear on someone else's terminal", () => {
    // The whole reason the face is `(o  o)` and not `( ･ᴗ･)`: an ambiguous-width
    // glyph measures one column here and two in a CJK-capable terminal, and the
    // art then runs through the frame's right edge.
    for (const line of logoLines(70, 20, theme)) {
      assert.equal(visibleWidth(line), [...line].length, line);
    }
  });

  it("themes the scenery apart from the animal and the name", () => {
    const seen = probe();
    logoLines(70, 20, seen);
    assert.equal(seen.tokens[0], "dim", "the branch recedes");
    assert.ok(seen.tokens.includes("muted"), "the sloth");
    assert.ok(seen.tokens.includes("accent"), "the name");
    assert.ok(!seen.tokens.some((token) => !["dim", "muted", "accent"].includes(token)), seen.tokens.join(","));
  });
});

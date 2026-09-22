/**
 * The mascot, drawn once per machine on the setup wizard.
 *
 * It earns its rows on exactly one screen. `/unsloth setup` is the only
 * overlay a user meets before they have decided this extension is theirs — it
 * runs automatically on a machine with no profile, and then only when asked —
 * so it is the one place where a logo is the point rather than an interruption.
 * Everywhere else the overlays are read at a glance, repeatedly, under a row
 * budget — the panel's list is sized against what its chrome already costs —
 * and six rows of sloth would be six model rows fewer every time it opens.
 *
 * Two things make it safe to put art in a terminal UI, and both are enforced
 * here rather than left to the caller:
 *
 *   - **It sheds, twice.** Beside the wordmark at full width, wordmark alone
 *     when the overlay is narrow, and nothing at all when the rows are needed
 *     by the screen itself. The wizard asks for what it can spare and gets what
 *     fits; a logo that pushes `⏎ save` off the bottom is a bug, not a logo.
 *   - **Every character is one column wide.** ASCII, box-drawing and block
 *     elements only — no emoji, no East Asian ambiguous glyphs. A face built
 *     from `･` or `◕` measures differently in every terminal, and the art then
 *     tears against the frame's right edge on somebody else's machine.
 *
 * Themed like everything else here: the branch recedes to `dim`, the sloth is
 * `muted`, the name is `accent`. No raw ANSI — the user's theme decides what
 * those mean, in light terminals as well as dark.
 */

export type LogoToken = "accent" | "muted" | "dim";

export interface LogoTheme {
  fg(color: LogoToken, text: string): string;
}

/** The sloth, hanging. The branch it hangs from is drawn to fit, below. */
const SLOTH: readonly string[] = [
  "  /\\        /\\",
  " (  )      (  )",
  "  \\  \\____/  /",
  "   \\ (o  o) /",
  "     \\ ~~ /",
];

/** `UNSLOTH` in half blocks. Three rows, because five would not shed anywhere. */
const WORDMARK: readonly string[] = [
  "█ █ █▄█ █▀▀ █   █▀█ ▀█▀ █ █",
  "█ █ █ █ ▀▀█ █   █ █  █  █▀█",
  "▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀ ▀▀▀  ▀  ▀ ▀",
];

/** The sloth row the wordmark starts level with, so it sits against the body. */
const WORDMARK_TOP = 1;
/** Columns between the two. Less than this and the sloth's claws touch the U. */
const GAP = 5;
/** Left margin — the same one every other line in the wizard uses. */
const INDENT = 2;

const SLOTH_WIDTH = Math.max(...SLOTH.map((line) => line.length));
const WORDMARK_WIDTH = Math.max(...WORDMARK.map((line) => line.length));

/** Columns each form needs, including the margin it is drawn at. */
export const LOGO_FULL_COLUMNS = INDENT + SLOTH_WIDTH + GAP + WORDMARK_WIDTH;
export const LOGO_SHORT_COLUMNS = INDENT + WORDMARK_WIDTH;

/** Rows each form costs — the sloth plus the branch above it. */
export const LOGO_FULL_ROWS = SLOTH.length + 1;
export const LOGO_SHORT_ROWS = WORDMARK.length;

/**
 * The logo, sized to what the caller can spare.
 *
 * `width` is the space inside the frame and `rows` the lines the screen can
 * give up. Either being short drops to the wordmark, and too little of either
 * returns nothing — an empty array, so the caller's layout simply closes up
 * rather than having to ask twice.
 */
export function logoLines(width: number, rows: number, theme: LogoTheme): string[] {
  if (width >= LOGO_FULL_COLUMNS && rows >= LOGO_FULL_ROWS) return full(theme);
  if (width >= LOGO_SHORT_COLUMNS && rows >= LOGO_SHORT_ROWS) return WORDMARK.map(wordmarkLine(theme));
  return [];
}

/**
 * The sloth with its name beside it, both hung from one branch.
 *
 * The branch runs the full width of the logo rather than stopping at the
 * animal. It is what makes the two halves one drawing: cut short, the sloth
 * hangs from a twig and the wordmark floats beside it with nothing holding it
 * up. Dim, because it is scenery — the eye should land on the sloth.
 */
function full(theme: LogoTheme): string[] {
  const pad = " ".repeat(INDENT);
  const branch = pad + theme.fg("dim", "─".repeat(SLOTH_WIDTH + GAP + WORDMARK_WIDTH));
  const body = SLOTH.map((art, row) => {
    const word = WORDMARK[row - WORDMARK_TOP];
    // No padding on a row with nothing beside it: trailing themed spaces are
    // invisible until something highlights the line, and then they are a box.
    if (word === undefined) return pad + theme.fg("muted", art);
    return pad + theme.fg("muted", art.padEnd(SLOTH_WIDTH)) + " ".repeat(GAP) + theme.fg("accent", word);
  });
  return [branch, ...body];
}

function wordmarkLine(theme: LogoTheme): (line: string) => string {
  return (line) => " ".repeat(INDENT) + theme.fg("accent", line);
}

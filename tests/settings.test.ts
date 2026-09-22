import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, describe, it } from "node:test";

import {
  PROFILE_FILENAME,
  autoUnloadOnExit,
  autostartEnabled,
  envBoolean,
  launchCommand,
  profileBaseUrl,
  readProfile,
} from "../src/settings.ts";

/** Pi resolves its agent directory from this, so a temp dir isolates the test. */
const AGENT_DIR_ENV = "PI_CODING_AGENT_DIR";

const OWNED_ENV = [
  AGENT_DIR_ENV,
  "UNSLOTH_AUTOSTART",
  "UNSLOTH_AUTO_UNLOAD_ON_EXIT",
  "UNSLOTH_LAUNCH_COMMAND",
];

let dir: string;
const saved = new Map<string, string | undefined>();

function writeProfile(contents: string): void {
  writeFileSync(join(dir, PROFILE_FILENAME), contents);
}

before(() => {
  for (const name of OWNED_ENV) saved.set(name, process.env[name]);
  dir = mkdtempSync(join(tmpdir(), "pi-unsloth-settings-"));
});

beforeEach(() => {
  for (const name of OWNED_ENV) delete process.env[name];
  process.env[AGENT_DIR_ENV] = dir;
  writeProfile("{}");
});

after(() => {
  for (const [name, value] of saved) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  rmSync(dir, { recursive: true, force: true });
});

describe("readProfile", () => {
  it("reads an object", () => {
    writeProfile(JSON.stringify({ baseUrl: "http://example:1234" }));
    assert.deepEqual(readProfile(), { baseUrl: "http://example:1234" });
  });

  it("survives a corrupt profile without throwing", () => {
    writeProfile("{ not json");
    assert.deepEqual(readProfile(), {});
    assert.equal(profileBaseUrl(), undefined);
  });

  it("survives a profile that is valid JSON but not an object", () => {
    writeProfile("[1, 2, 3]");
    assert.deepEqual(readProfile(), {});
  });

  it("survives no profile at all", () => {
    process.env[AGENT_DIR_ENV] = join(dir, "does-not-exist");
    assert.deepEqual(readProfile(), {});
  });
});

describe("envBoolean", () => {
  it("accepts the unambiguous spellings", () => {
    for (const value of ["1", "true", "YES", "on"]) {
      process.env["UNSLOTH_AUTOSTART"] = value;
      assert.equal(envBoolean("UNSLOTH_AUTOSTART"), true, value);
    }
    for (const value of ["0", "false", "NO", "off"]) {
      process.env["UNSLOTH_AUTOSTART"] = value;
      assert.equal(envBoolean("UNSLOTH_AUTOSTART"), false, value);
    }
  });

  it("treats an empty or unrecognised value as unset, never as false", () => {
    process.env["UNSLOTH_AUTOSTART"] = "";
    assert.equal(envBoolean("UNSLOTH_AUTOSTART"), undefined);
    process.env["UNSLOTH_AUTOSTART"] = "maybe";
    assert.equal(envBoolean("UNSLOTH_AUTOSTART"), undefined);
  });
});

describe("policies — environment beats profile beats default", () => {
  it("autostart defaults on: it is the feature that replaces pil", () => {
    assert.equal(autostartEnabled(), true);
  });

  it("auto-unload defaults off: another window may still want the model", () => {
    assert.equal(autoUnloadOnExit(), false);
  });

  it("takes both policies from the profile", () => {
    writeProfile(JSON.stringify({ autostart: false, autoUnloadOnExit: true }));
    assert.equal(autostartEnabled(), false);
    assert.equal(autoUnloadOnExit(), true);
  });

  it("lets the environment override the profile", () => {
    writeProfile(JSON.stringify({ autostart: false, autoUnloadOnExit: true }));
    process.env["UNSLOTH_AUTOSTART"] = "1";
    process.env["UNSLOTH_AUTO_UNLOAD_ON_EXIT"] = "0";
    assert.equal(autostartEnabled(), true);
    assert.equal(autoUnloadOnExit(), false);
  });

  it("ignores a profile value of the wrong type", () => {
    writeProfile(JSON.stringify({ autostart: "no" }));
    assert.equal(autostartEnabled(), true);
  });
});

describe("launchCommand", () => {
  it("defaults to the binary Unsloth Studio installs", () => {
    assert.deepEqual(launchCommand(), ["unsloth-studio"]);
  });

  it("splits a string from the profile into argv", () => {
    writeProfile(JSON.stringify({ launchCommand: "flatpak run ai.unsloth.Studio" }));
    assert.deepEqual(launchCommand(), ["flatpak", "run", "ai.unsloth.Studio"]);
  });

  it("takes an argv array verbatim, so an argument may contain spaces", () => {
    writeProfile(JSON.stringify({ launchCommand: ["/opt/my studio/bin/start", "--api-only"] }));
    assert.deepEqual(launchCommand(), ["/opt/my studio/bin/start", "--api-only"]);
  });

  it("lets the environment override the profile", () => {
    writeProfile(JSON.stringify({ launchCommand: "from-profile" }));
    process.env["UNSLOTH_LAUNCH_COMMAND"] = "from-env --flag";
    assert.deepEqual(launchCommand(), ["from-env", "--flag"]);
  });

  it("falls back to the default when the configured value is empty", () => {
    writeProfile(JSON.stringify({ launchCommand: "   " }));
    assert.deepEqual(launchCommand(), ["unsloth-studio"]);
  });
});

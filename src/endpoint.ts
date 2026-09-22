/**
 * Is the configured endpoint this machine?
 *
 * Three things this extension does act on the computer it is running on rather
 * than on the server it is talking to: starting Unsloth Studio, stopping
 * `llama-server` (the only way VRAM is actually freed — there is no endpoint
 * for it), and reading `/sys/class/drm` to see which GPU has a monitor plugged
 * in. Every one of them is wrong when the server is somewhere else, and the
 * worst of them is wrong *destructively*: on 2026-09-21 a session pointed at a
 * stub server still terminated the real inference process on the reference box,
 * because the unload's second half never looked at the endpoint it had been
 * configured with.
 *
 * So locality is decided here, once, and it is a property of the address rather
 * than a setting anyone has to remember to tick.
 *
 * Nothing here resolves DNS or touches the network — the answer is needed on
 * paths that must not block, and a rule that sometimes takes a second is a rule
 * that gets called from the wrong place eventually. A hostname this machine
 * cannot recognise *by name* is therefore treated as remote, which costs a
 * feature and never costs someone else's model.
 */

import { hostname, networkInterfaces } from "node:os";

import { DEFAULT_BASE_URL } from "./api/client.ts";

/** Port assumed when an address names a host and nothing else. */
export const DEFAULT_PORT = 8888;

/**
 * What this machine answers to: every address on every interface, plus its own
 * name. Injected in tests, because a rule that can only be exercised on the
 * machine that wrote it is not a tested rule.
 */
export interface Machine {
  addresses: readonly string[];
  hostname: string;
}

/** Strip an IPv6 zone (`fe80::1%eth0`) and the brackets a URL puts round it. */
function bareHost(value: string): string {
  return value.replace(/^\[|\]$/g, "").split("%")[0]!.trim().toLowerCase();
}

/** This machine's own addresses and name. */
export function thisMachine(): Machine {
  const addresses: string[] = [];
  try {
    for (const entries of Object.values(networkInterfaces())) {
      for (const entry of entries ?? []) addresses.push(bareHost(entry.address));
    }
  } catch {
    // No interface list — every address then falls through to the loopback
    // test, which is the conservative answer.
  }
  let name = "";
  try {
    name = hostname();
  } catch {
    // Same.
  }
  return { addresses, hostname: name.toLowerCase() };
}

/**
 * Loopback, by the address alone.
 *
 * `0.0.0.0` and `::` are included: as a *destination* they reach this machine,
 * and a user who typed the address their server binds to is talking to their
 * own box.
 */
export function isLoopbackHost(host: string): boolean {
  const value = bareHost(host);
  if (value === "localhost" || value.endsWith(".localhost")) return true;
  if (value === "::1" || value === "::" || value === "0:0:0:0:0:0:0:1") return true;
  if (value === "0.0.0.0") return true;
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(value);
}

/** The host of a base URL, or `undefined` when it is not a URL at all. */
export function endpointHost(baseUrl: string): string | undefined {
  try {
    const host = new URL(baseUrl).hostname;
    return host === "" ? undefined : bareHost(host);
  } catch {
    return undefined;
  }
}

/** `192.168.1.40:8888` — what the UI shows, and what a message names. */
export function endpointLabel(baseUrl: string): string {
  return baseUrl.replace(/^https?:\/\//, "").replace(/\/+$/, "");
}

/**
 * Is `baseUrl` served by the machine this code is running on?
 *
 * An address that is not a URL at all answers **false**: it cannot be talked to
 * either way, and "remote" is the reading that touches nothing local.
 */
export function isLocalEndpoint(baseUrl: string, machine: Machine = thisMachine()): boolean {
  const host = endpointHost(baseUrl);
  if (host === undefined) return false;
  if (isLoopbackHost(host)) return true;
  if (machine.addresses.includes(host)) return true;
  if (host === machine.hostname) return true;
  // `gpubox` and `gpubox.lan` are the same machine talking about itself; the
  // first label is compared so a search domain does not make a box foreign.
  const label = machine.hostname.split(".")[0];
  return label !== undefined && label !== "" && host.split(".")[0] === label;
}

/** Do these two addresses name the same endpoint? Used to age out a profile. */
export function sameEndpoint(a: string | undefined, b: string | undefined): boolean {
  if (a === undefined || b === undefined) return false;
  return normaliseEndpoint(a) === normaliseEndpoint(b);
}

/**
 * `192.168.1.40` → `http://192.168.1.40:8888`.
 *
 * What the wizard's `e` accepts: a host, a `host:port`, or a full URL when the
 * server sits behind TLS or a proxy. `undefined` when the text cannot be read
 * as an address at all, which is the wizard's cue to keep the field open rather
 * than store a guess.
 */
export function normaliseEndpoint(text: string): string | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;

  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    return undefined;
  }
  if (url.hostname === "") return undefined;
  if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
  // A bare host means the documented default port, but only over plain HTTP:
  // an `https://` address is a proxy in front of something, and 443 is that
  // proxy's business rather than ours.
  if (url.port === "" && url.protocol === "http:") url.port = String(DEFAULT_PORT);

  const path = url.pathname.replace(/\/+$/, "");
  return `${url.protocol}//${url.host}${path}`;
}

/** The address the wizard opens on when there is nothing configured yet. */
export function defaultEndpoint(): string {
  return DEFAULT_BASE_URL;
}

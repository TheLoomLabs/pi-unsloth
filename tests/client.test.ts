import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { after, before, describe, it } from "node:test";

import {
  DEFAULT_BASE_URL,
  UnslothApiError,
  UnslothClient,
  UnslothTimeoutError,
  UnslothUnreachableError,
  checkHealth,
  isHealthy,
  resolveEndpoint,
} from "../src/api/client.ts";

/** A stand-in Unsloth server, so these tests need no real one. */
interface Stub {
  url: string;
  close: () => Promise<void>;
  /** Authorization header of the last request. */
  lastAuth: string | undefined;
  /** Path + query of the last request. */
  lastUrl: string | undefined;
}

/** An ephemeral port that is guaranteed free and not on Node's blocked list. */
async function freePort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (typeof address === "string" || address === null) throw new Error("no port");
  const { port } = address;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}

async function startStub(handler: (path: string) => { status: number; body: string }): Promise<Stub> {
  const stub: Partial<Stub> = {};
  const server: Server = createServer((req, res) => {
    stub.lastAuth = req.headers.authorization;
    stub.lastUrl = req.url;
    const { status, body } = handler(req.url ?? "/");
    res.writeHead(status, { "Content-Type": "application/json" });
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (typeof address === "string" || address === null) throw new Error("no port");
  stub.url = `http://127.0.0.1:${address.port}`;
  stub.close = () => new Promise<void>((resolve) => server.close(() => resolve()));
  return stub as Stub;
}

describe("resolveEndpoint", () => {
  it("prefers an explicit option over everything else", () => {
    assert.equal(resolveEndpoint({ baseUrl: "http://explicit:1" }).baseUrl, "http://explicit:1");
  });

  it("reads UNSLOTH_BASE_URL and strips a trailing slash", () => {
    process.env.UNSLOTH_BASE_URL = "http://from-env:9999/";
    try {
      assert.equal(resolveEndpoint().baseUrl, "http://from-env:9999");
    } finally {
      delete process.env.UNSLOTH_BASE_URL;
    }
  });

  it("falls back to the documented local default", () => {
    assert.equal(resolveEndpoint().baseUrl, DEFAULT_BASE_URL);
  });
});

describe("UnslothClient", () => {
  let ok: Stub;

  before(async () => {
    ok = await startStub((path) =>
      path.startsWith("/api/health")
        ? { status: 200, body: '{"status":"ok"}' }
        : path.startsWith("/api/boom")
          ? { status: 503, body: '{"reason":"model is still loading"}' }
          : path.startsWith("/api/denied")
            ? { status: 401, body: '{"detail":"bad key"}' }
            : path.startsWith("/api/empty")
              ? { status: 204, body: "" }
              : { status: 200, body: '{"echo":true}' },
    );
  });

  after(async () => {
    await ok.close();
  });

  it("sends a bearer token when it has one", async () => {
    await new UnslothClient({ baseUrl: ok.url, apiKey: "secret-key" }).get("/api/thing");
    assert.equal(ok.lastAuth, "Bearer secret-key");
  });

  it("omits the Authorization header for anonymous calls", async () => {
    await new UnslothClient({ baseUrl: ok.url, apiKey: "secret-key" }).request("/api/health", {
      anonymous: true,
    });
    assert.equal(ok.lastAuth, undefined);
  });

  it("appends query parameters and drops undefined ones", async () => {
    await new UnslothClient({ baseUrl: ok.url }).get("/api/thing", {
      query: { keep: 1, skip: undefined },
    });
    assert.equal(ok.lastUrl, "/api/thing?keep=1");
  });

  it("carries the server's own reason on an error", async () => {
    await assert.rejects(
      () => new UnslothClient({ baseUrl: ok.url }).get("/api/boom"),
      (error: unknown) => {
        assert.ok(error instanceof UnslothApiError);
        assert.equal(error.status, 503);
        assert.equal(error.reason, "model is still loading");
        return true;
      },
    );
  });

  it("recognises an auth failure", async () => {
    await assert.rejects(
      () => new UnslothClient({ baseUrl: ok.url }).get("/api/denied"),
      (error: unknown) => {
        assert.ok(error instanceof UnslothApiError);
        assert.equal(error.isAuthError, true);
        assert.equal(error.reason, "bad key");
        return true;
      },
    );
  });

  it("treats an empty body as no value rather than a parse error", async () => {
    assert.equal(await new UnslothClient({ baseUrl: ok.url }).get("/api/empty"), undefined);
  });

  it("reports an unreachable server with the real cause, not 'fetch failed'", async () => {
    const port = await freePort();
    await assert.rejects(
      () => new UnslothClient({ baseUrl: `http://127.0.0.1:${port}` }).get("/api/system"),
      (error: unknown) => {
        assert.ok(error instanceof UnslothUnreachableError);
        assert.match(error.message, /ECONNREFUSED|ECONNRESET/);
        return true;
      },
    );
  });

  it("times out on its own budget", async () => {
    const slow = createServer(() => {
      /* never responds */
    });
    await new Promise<void>((resolve) => slow.listen(0, "127.0.0.1", resolve));
    const address = slow.address();
    if (typeof address === "string" || address === null) throw new Error("no port");
    try {
      const started = Date.now();
      await assert.rejects(
        () => new UnslothClient({ baseUrl: `http://127.0.0.1:${address.port}` }).get("/api/system", { timeoutMs: 200 }),
        (error: unknown) => error instanceof UnslothTimeoutError,
      );
      assert.ok(Date.now() - started < 2000, "timeout should fire near its budget");
    } finally {
      slow.closeAllConnections();
      await new Promise<void>((resolve) => slow.close(() => resolve()));
    }
  });

  it("rejects promptly when the caller aborts", async () => {
    const slow = createServer(() => {
      /* never responds */
    });
    await new Promise<void>((resolve) => slow.listen(0, "127.0.0.1", resolve));
    const address = slow.address();
    if (typeof address === "string" || address === null) throw new Error("no port");
    try {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 30);
      const started = Date.now();
      await assert.rejects(
        () =>
          new UnslothClient({ baseUrl: `http://127.0.0.1:${address.port}` }).get("/api/system", {
            timeoutMs: 30_000,
            signal: controller.signal,
          }),
        (error: unknown) => (error as Error).name === "AbortError",
      );
      assert.ok(Date.now() - started < 2000, "abort should reject immediately, not at the timeout");
    } finally {
      slow.closeAllConnections();
      await new Promise<void>((resolve) => slow.close(() => resolve()));
    }
  });
});

describe("checkHealth", () => {
  it("reports a reachable server as up", async () => {
    const stub = await startStub(() => ({ status: 200, body: "{}" }));
    try {
      const result = await checkHealth(new UnslothClient({ baseUrl: stub.url }));
      assert.equal(result.state, "up");
      assert.ok(result.latencyMs >= 0);
    } finally {
      await stub.close();
    }
  });

  it("returns false within the timeout when the server is down, without throwing", async () => {
    const port = await freePort();
    const started = Date.now();
    const healthy = await isHealthy(new UnslothClient({ baseUrl: `http://127.0.0.1:${port}` }), {
      timeoutMs: 800,
    });
    assert.equal(healthy, false);
    assert.ok(Date.now() - started < 1500);
  });

  it("distinguishes a rejected key from an unreachable server", async () => {
    const stub = await startStub(() => ({ status: 403, body: '{"reason":"forbidden"}' }));
    try {
      const result = await checkHealth(new UnslothClient({ baseUrl: stub.url }));
      assert.equal(result.state, "unauthorized");
      assert.equal(result.detail, "forbidden");
    } finally {
      await stub.close();
    }
  });
});

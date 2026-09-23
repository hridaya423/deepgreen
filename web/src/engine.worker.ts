/*
 * Model init off the main thread: fetches the release bundle, verifies
 * digests, and runs the per-word key precompute here so the page never
 * blocks. Computed keys persist in Cache API (keyed by the certified
 * weights sha256) so repeat visits skip the ~2.5G-MAC loop entirely.
 * Everything returns via transferable postMessage — zero copies.
 */

import {
  collectBuffers,
  fetchBundle,
  prepareEngine,
  sha256hex,
  type EngineParts,
} from "./engine.js";

const CACHE_NAME = "wordle-engine-v1";

const keysUrl = (weightsSha: string): string => `/engine-keys/${weightsSha}`;

interface CachedKeys {
  count: number;
  width: number;
  keys: Float32Array;
  keyBias: Float32Array;
}

async function readCachedKeys(weightsSha: string): Promise<CachedKeys | null> {
  const hit = await (await caches.open(CACHE_NAME)).match(keysUrl(weightsSha));
  if (!hit) return null;
  const buf = await hit.arrayBuffer();
  if (buf.byteLength < 8) return null;
  const head = new Uint32Array(buf, 0, 2);
  const [count, width] = [head[0]!, head[1]!];
  if (buf.byteLength !== 8 + count * 4 + count * width * 4) return null;
  return {
    count,
    width,
    keyBias: new Float32Array(buf, 8, count),
    keys: new Float32Array(buf, 8 + count * 4, count * width),
  };
}

async function storeKeys(
  weightsSha: string,
  parts: EngineParts,
): Promise<void> {
  const count = parts.words.length;
  const buf = new ArrayBuffer(8 + count * 4 + count * parts.width * 4);
  new Uint32Array(buf, 0, 2).set([count, parts.width]);
  new Float32Array(buf, 8, count).set(parts.keyBias);
  new Float32Array(buf, 8 + count * 4, count * parts.width).set(parts.keys);
  const cache = await caches.open(CACHE_NAME);
  // Evict keys for superseded payloads — only the current weights' entry is useful.
  for (const request of await cache.keys()) {
    if (request.url !== keysUrl(weightsSha)) await cache.delete(request);
  }
  await cache.put(keysUrl(weightsSha), new Response(buf));
}

const scope = self as unknown as {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
};

scope.onmessage = async (event: MessageEvent): Promise<void> => {
  const { base, version } = event.data as { base: string; version: string };
  try {
    const bundle = await fetchBundle(base, version);
    const weightsSha = await sha256hex(bundle.weights);
    let cached: CachedKeys | null = null;
    try {
      cached = await readCachedKeys(weightsSha);
    } catch {
      /* cache unavailable — compute */
    }
    const parts = await prepareEngine(bundle, cached ?? undefined);
    if (!cached) {
      try {
        await storeKeys(weightsSha, parts);
      } catch {
        /* cache is best-effort */
      }
    }
    scope.postMessage({ ok: true, parts }, collectBuffers(parts));
  } catch (error) {
    scope.postMessage({
      ok: false,
      message: error instanceof Error ? error.message : "model load failed",
    });
  }
};

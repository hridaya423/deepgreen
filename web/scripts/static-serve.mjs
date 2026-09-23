/* Serve web/dist exactly as a static host would (Vercel, Pages, nginx). */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(
  join(dirname(fileURLToPath(import.meta.url)), "..", "dist"),
);
const port = Number(process.env.PORT ?? 3101);
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".txt": "text/plain; charset=utf-8",
  ".safetensors": "application/octet-stream",
};

createServer(async (req, res) => {
  let pathname = decodeURIComponent(
    new URL(req.url ?? "/", "http://local").pathname,
  );
  if (pathname.endsWith("/")) pathname += "index.html";
  const file = join(root, normalize(pathname));
  if (!file.startsWith(root)) {
    res.writeHead(404).end("not found");
    return;
  }
  try {
    const data = await readFile(file);
    res.writeHead(200, {
      "content-type": TYPES[extname(file)] ?? "application/octet-stream",
      ...(pathname.startsWith("/model/")
        ? { "cache-control": "public, max-age=31536000, immutable" }
        : {}),
    });
    res.end(data);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`serving dist on http://127.0.0.1:${port}`);
});

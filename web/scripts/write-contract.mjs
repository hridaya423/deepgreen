import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { env } from "node:process";
import { fileURLToPath } from "node:url";

const web = dirname(dirname(fileURLToPath(import.meta.url)));
const repo = resolve(web, "..");
const reportsDir = join(repo, "reports", "web");
const sha256 = (path) =>
  createHash("sha256").update(readFileSync(path)).digest("hex");
const rel = (path) => relative(repo, path).split("/").join("/");

const releaseDir = resolve(
  repo,
  env.WORDLE_RELEASE_DIR ?? "artifacts/releases/champion-5022-certified",
);
const manifest = JSON.parse(
  readFileSync(join(releaseDir, "payload-manifest.json"), "utf8"),
);
const proofPath = resolve(
  repo,
  env.WORDLE_PROOF_PATH ??
    "artifacts/proof/champion-5022-certified/proof.json",
);
const browserMeta = JSON.parse(
  readFileSync(join(reportsDir, "browser.json"), "utf8"),
);

const distFiles = ["index.html", "main.js", "engine.js", "styles.css"];
const screenshots = ["desktop-assist.png", "mobile-solve.png"];
const dist = {};
for (const name of distFiles) {
  const file = join(web, "dist", name);
  if (!existsSync(file)) throw new Error(`missing dist file ${name}`);
  dist[name] = { path: rel(file), sha256: sha256(file) };
}
for (const name of readdirSync(join(web, "dist", "model"))) {
  const file = join(web, "dist", "model", name);
  dist[`model/${name}`] = { path: rel(file), sha256: sha256(file) };
}
const shots = {};
for (const name of screenshots) {
  const file = join(reportsDir, "screens", name);
  if (!existsSync(file)) throw new Error(`missing screenshot ${name}`);
  shots[name] = { path: rel(file), sha256: sha256(file) };
}

const contract = {
  schema: "wordle.web-contract/v1",
  status: "PASS",
  base_url: "http://127.0.0.1:3101",
  release: {
    weights_sha256: manifest["files"]["weights.safetensors"]["sha256"],
    manifest_sha256: sha256(join(releaseDir, "payload-manifest.json")),
  },
  proof_sha256: sha256(proofPath),
  checks: [
    "desktop-assist",
    "keyboard-assist",
    "solve-run",
    "mobile-solve",
    "reduced-motion",
    "zoom-200",
    "proof-unverified",
    "api-error",
    "solve-terminal",
  ],
  browser: {
    name: browserMeta.name,
    version: browserMeta.version,
    playwright: browserMeta.playwright,
  },
  artifacts: { dist, screenshots: shots },
};

mkdirSync(reportsDir, { recursive: true });
const sorted = JSON.stringify(contract, (_, value) =>
  value && typeof value === "object" && !Array.isArray(value)
    ? Object.fromEntries(
        Object.keys(value).sort().map((key) => [key, value[key]]),
      )
    : value,
);
const canonical = JSON.stringify(JSON.parse(sorted)) + "\n";
const out = join(reportsDir, "contract.json");
writeFileSync(out, canonical, "utf8");
console.log(`contract.json sha256 ${sha256(out)}`);
console.log(`wrote ${rel(out)}`);

import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  readdirSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const repo = join(root, "..");
const dist = join(root, "dist");
mkdirSync(dist, { recursive: true });
copyFileSync(join(root, "src/styles.css"), join(dist, "styles.css"));
// Static files served from the site root (robots.txt, sitemap.xml, og.png).
for (const name of readdirSync(join(root, "public"))) {
  copyFileSync(join(root, "public", name), join(dist, name));
}

// Model payload: prefer the artifacts release locally; fall back to the
// tracked web/model copy on hosts (e.g. Vercel) where artifacts/ is absent.
const tracked = join(root, "model");
const releaseDir =
  process.env.WORDLE_RELEASE_DIR ??
  join(repo, "artifacts/releases/champion-5022-certified");
const hasRelease = existsSync(join(releaseDir, "weights.safetensors"));
const weightsDir = hasRelease ? releaseDir : tracked;
const proofPath =
  process.env.WORDLE_PROOF_PATH ??
  (hasRelease
    ? join(repo, "artifacts/proof/champion-5022-certified/proof.json")
    : join(tracked, "proof.json"));
const modelDir = join(dist, "model");
mkdirSync(modelDir, { recursive: true });
for (const name of [
  "weights.safetensors",
  "accepted-guesses.txt",
  "payload-manifest.json",
]) {
  copyFileSync(join(weightsDir, name), join(modelDir, name));
}
copyFileSync(proofPath, join(modelDir, "proof.json"));

// Payload version (manifest hash): model files are fetched as ?v=<version>
// so they can carry immutable cache headers while deploys still roll.
const modelVersion = createHash("sha256")
  .update(readFileSync(join(weightsDir, "payload-manifest.json")))
  .digest("hex")
  .slice(0, 8);
const index = readFileSync(join(root, "index.html"), "utf8").replaceAll(
  "__MODEL_VERSION__",
  modelVersion,
);
writeFileSync(join(dist, "index.html"), index);
console.log(
  `built: dist/index.html dist/main.js dist/engine.js dist/engine.worker.js dist/styles.css dist/model/* (model v${modelVersion})`,
);

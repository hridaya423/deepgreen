/* Captures the Return path result screen states for design review.
   Serves web/dist statically, drives real engine solves (plus one
   wrapped-engine fixture for the six-turn edge), and saves PNGs to
   reports/web/screens/return-path/. */
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(join(here, ".."));
const OUT = resolve(join(web, "..", "reports", "web", "screens", "return-path"));
const PORT = 3199;
const BASE = `http://127.0.0.1:${PORT}`;

const server = spawn("node", [join(here, "static-serve.mjs")], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: "ignore",
});
await new Promise((r) => setTimeout(r, 800));

const SIX_TURN = {
  schema: "wordle.solve/v1",
  mode: "normal",
  word: "crane",
  rows: [
    { guess: "stare", marks: "bbgyg" },
    { guess: "mound", marks: "bbbgb" },
    { guess: "climb", marks: "gbbbb" },
    { guess: "fjord", marks: "bbbyb" },
    { guess: "vexes", marks: "bybbb" },
    { guess: "wrung", marks: "bgbgb" },
  ],
  turns: 6,
  consistent_counts: [62, 12, 6, 3, 2, 1],
  vocabulary_size: 5022,
  outcome: "UNSOLVED",
  certification: "UNVERIFIED",
  identity: { proof_status: "UNVERIFIED" },
};

async function stubSolve(page, fixture) {
  const source = readFileSync(join(web, "dist", "engine.js"), "utf8");
  await page.route("**/engine-impl.js", (route) =>
    route.fulfill({ contentType: "text/javascript", body: source }),
  );
  await page.route("**/engine.js", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: `import * as real from "./engine-impl.js";
export const EngineError = real.EngineError;
export const fetchBundle = real.fetchBundle;
export const fetchProof = real.fetchProof;
export async function loadEngine(bundle) {
  const engine = await real.loadEngine(bundle);
  const fixture = globalThis.__solveFixture;
  if (!fixture) return engine;
  return { ...engine, solve: () => fixture };
}`,
    }),
  );
  await page.addInitScript((data) => {
    globalThis.__solveFixture = data;
  }, fixture);
}

async function solved(page, word = "crane") {
  await page.goto(`${BASE}/`);
  await page.getByRole("textbox", { name: "Your word", exact: true }).fill(word);
  await page.getByRole("button", { name: "Solve my word", exact: true }).click();
  await page.locator(".return-path-footer").waitFor({ timeout: 20_000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(200);
}

function contrast(fg, bg) {
  const lum = (c) => {
    const [r, g, b] = c.map((v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const parse = (s) => {
    const m = s.match(/[\d.]+/g).map(Number);
    return m.slice(0, 3);
  };
  const a = lum(parse(fg));
  const b = lum(parse(bg));
  return ((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)).toFixed(2);
}

async function measureContrast(page) {
  return page.evaluate(() => {
    const pick = (sel) => {
      const el = document.querySelector(sel);
      return el ? getComputedStyle(el) : null;
    };
    return {
      countValue: pick(".return-path-count-value")?.color,
      countUnit: pick(".return-path-count-unit")?.color,
      tileInk: pick(".return-path-result .tile")?.color,
      tileBg: pick(".return-path-result .tile")?.backgroundColor,
      bg: getComputedStyle(document.body).backgroundColor,
      ink: getComputedStyle(document.body).color,
      turnBorder: pick(".return-path-turn")?.borderColor,
      pathStroke: pick(".trace-spine")?.stroke,
      leaderStroke: pick(".trace-leader")?.stroke,
    };
  });
}

async function snap(page, name) {
  await page.screenshot({ path: join(OUT, `${name}.png`), fullPage: true, animations: "disabled" });
  console.log("saved", name);
}

try {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const ctx = async (opts) => browser.newContext({ baseURL: BASE, ...opts });

  const measures = {};

  // desktop dark (reference comparison)
  {
    const c = await ctx({
      viewport: { width: 1586, height: 992 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, "desktop-dark");
    const m = await measureContrast(page);
    m.inkOnBg = contrast(m.ink, m.bg);
    m.countValueOnBg = contrast(m.countValue, m.bg);
    m.countUnitOnBg = contrast(m.countUnit, m.bg);
    m.tileInkOnTile = contrast(m.tileInk, m.tileBg);
    m.pathOnBg = contrast(m.pathStroke, m.bg);
    m.leaderOnBg = contrast(m.leaderStroke, m.bg);
    measures["desktop-dark"] = m;
    await c.close();
  }

  // desktop light
  {
    const c = await ctx({
      viewport: { width: 1586, height: 992 },
      colorScheme: "light",
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, "desktop-light");
    const m = await measureContrast(page);
    m.inkOnBg = contrast(m.ink, m.bg);
    m.countValueOnBg = contrast(m.countValue, m.bg);
    m.countUnitOnBg = contrast(m.countUnit, m.bg);
    m.tileInkOnTile = contrast(m.tileInk, m.tileBg);
    m.pathOnBg = contrast(m.pathStroke, m.bg);
    m.leaderOnBg = contrast(m.leaderStroke, m.bg);
    measures["desktop-light"] = m;
    await c.close();
  }

  // laptop
  {
    const c = await ctx({
      viewport: { width: 1440, height: 900 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, "laptop");
    await c.close();
  }

  // tablet
  {
    const c = await ctx({
      viewport: { width: 834, height: 1112 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, "tablet");
    await c.close();
  }

  // mobile dark + light
  for (const scheme of ["dark", "light"]) {
    const c = await ctx({
      viewport: { width: 375, height: 812 },
      colorScheme: scheme,
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, `mobile-${scheme}`);
    await c.close();
  }

  // narrow
  {
    const c = await ctx({
      viewport: { width: 320, height: 568 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, "narrow");
    await c.close();
  }

  // short viewport
  {
    const c = await ctx({
      viewport: { width: 1366, height: 560 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page);
    await snap(page, "short");
    await c.close();
  }

  // one turn
  {
    const c = await ctx({
      viewport: { width: 1586, height: 992 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page, "tares");
    await snap(page, "one-turn");
    await c.close();
  }

  // six-turn unsolved (engine wrapper fixture), desktop + mobile
  for (const [name, viewport] of [
    ["six-turn", { width: 1586, height: 992 }],
    ["unsolved", { width: 375, height: 812 }],
  ] as const) {
    const c = await ctx({ viewport, colorScheme: "dark" });
    const page = await c.newPage();
    await stubSolve(page, SIX_TURN);
    await solved(page);
    await snap(page, name);
    await c.close();
  }

  // selected turn
  {
    const c = await ctx({
      viewport: { width: 1586, height: 992 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await solved(page);
    await page
      .getByRole("button", { name: "Inspect guess 2, DECAL", exact: true })
      .click();
    await page.waitForTimeout(120);
    await snap(page, "selected-turn");
    await c.close();
  }

  // mid-reveal
  {
    const c = await ctx({
      viewport: { width: 1586, height: 992 },
      colorScheme: "dark",
    });
    const page = await c.newPage();
    await page.clock.pauseAt(Date.now());
    await page.goto(`${BASE}/`);
    await page
      .getByRole("textbox", { name: "Your word", exact: true })
      .fill("crane");
    await page
      .getByRole("button", { name: "Solve my word", exact: true })
      .click();
    await page
      .locator(".return-path-result")
      .waitFor({ timeout: 20_000 });
    await page.clock.runFor(950);
    await page.evaluate(() => document.fonts.ready);
    await snap(page, "revealing");
    await c.close();
  }

  writeFileSync(
    join(OUT, "contrast.json"),
    JSON.stringify(measures, null, 2) + "\n",
  );
  console.log("contrast:", JSON.stringify(measures, null, 2));
  await browser.close();
} finally {
  server.kill();
}

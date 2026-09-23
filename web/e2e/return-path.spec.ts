import { expect, test, type Page } from "@playwright/test";
import { AxeBuilder } from "@axe-core/playwright";
import { readFileSync } from "node:fs";
import path from "node:path";

// Real "crane" trajectory from web/testdata/parity.json — matches the approved
// reference image (tares/decal/capon/crane, counts 62/6/1/1, CERTIFIED).
const referenceResult = {
  schema: "wordle.solve/v1",
  mode: "normal",
  word: "crane",
  rows: [
    { guess: "tares", marks: "byyyb" },
    { guess: "decal", marks: "byyyb" },
    { guess: "capon", marks: "gybby" },
    { guess: "crane", marks: "ggggg" },
  ],
  turns: 4,
  consistent_counts: [62, 6, 1, 1],
  vocabulary_size: 5022,
  outcome: "SOLVED",
  certification: "CERTIFIED",
  identity: { proof_status: "VERIFIED" },
};

async function showResult(page: Page, word = "crane") {
  await page.goto("/");
  await page.getByRole("textbox", { name: "Your word", exact: true }).fill(word);
  await page.getByRole("button", { name: "Solve my word", exact: true }).click();
  await expect(page.locator(".return-path-result .result")).toBeVisible({
    timeout: 15_000,
  });
}

// Test seam: substitute engine.js with a wrapper whose solve() returns a
// controlled fixture. Rows/counts the policy cannot produce (unsolved, missing
// counts) need this; real solves cover everything else.
async function stubSolve(page: Page, fixture: unknown) {
  const source = readFileSync(path.resolve("dist/engine.js"), "utf8");
  await page.route("**/engine-impl.js", (route) =>
    route.fulfill({ contentType: "text/javascript", body: source }),
  );
  await page.route("**/engine.js", (route) =>
    route.fulfill({
      contentType: "text/javascript",
      body: `import * as real from "./engine-impl.js";
export * from "./engine-impl.js";
const stubbed = (engine) => {
  const fixture = globalThis.__solveFixture;
  if (!fixture) return engine;
  return { ...engine, solve: () => fixture };
};
export function createEngine(parts) {
  return stubbed(real.createEngine(parts));
}
export async function loadEngine(bundle) {
  return stubbed(await real.loadEngine(bundle));
}`,
    }),
  );
  await page.addInitScript((data) => {
    (globalThis as Record<string, unknown>).__solveFixture = data;
  }, fixture);
}

test("result rows show each guess, its count, and the return segment", async ({ page }) => {
  await page.setViewportSize({ width: 1586, height: 992 });
  await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
  await showResult(page);
  const rows = page.locator(".return-path-row");
  await expect(rows).toHaveCount(4);
  await expect(rows.nth(0).locator(".tile")).toHaveText(["T", "A", "R", "E", "S"]);
  await expect(rows.nth(1).locator(".tile")).toHaveText(["D", "E", "C", "A", "L"]);
  await expect(rows.nth(2)).toContainText("1 possible word");
  await expect(rows.nth(2)).not.toContainText("Confirmed");
  await expect(rows.nth(3)).toContainText("Confirmed");
  await expect(page.locator(".return-path-origin")).toContainText("5,022");
  await expect(page.locator(".return-path-result .proof-rail")).toHaveCount(0);
  await expect(page.locator(".return-path-result .trace-return"))
    .toHaveAttribute("d", /M.+L/);
});

test("turn selection inspects without replaying the solve", async ({ page }) => {
  await showResult(page);
  let network = 0;
  page.on("request", (request) => {
    if (/(api|model)\//.test(request.url())) network += 1;
  });
  const turn = page.getByRole("button", { name: "Inspect guess 2, DECAL", exact: true });
  await turn.focus();
  await turn.press("Space");
  await expect(turn).toHaveAttribute("aria-pressed", "true");
  await expect(turn).toBeFocused();
  await expect(page.locator(".return-path-selection"))
    .toContainText("62 → 6 possible words");
  await expect(page.locator(".return-path-row")).toHaveCount(4);
  expect(network).toBe(0);
});

test("one-turn result resets to the existing composer", async ({ page }) => {
  await showResult(page, "tares");
  await expect(page.locator(".return-path-row")).toHaveCount(1);
  await expect(page.getByText("One guess to TARES.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Another word", exact: true }).click();
  const input = page.getByRole("textbox", { name: "Your word", exact: true });
  await expect(input).toHaveValue("");
  await expect(input).toBeFocused();
  await expect(page.locator(".return-path-result")).toHaveCount(0);
});

test("six-turn unsuccessful result has no success decoration", async ({ page }) => {
  await stubSolve(page, {
    ...referenceResult,
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
    outcome: "UNSOLVED",
    certification: "UNVERIFIED",
    identity: { proof_status: "UNVERIFIED" },
  });
  await showResult(page);
  await expect(page.locator(".return-path-row")).toHaveCount(6);
  await expect(page.locator(".return-path-result .result")).toContainText("Not solved");
  await expect(page.locator(".return-path-result .trace-return")).toHaveCount(0);
  await expect(page.locator(".return-path-result")).not.toContainText("Confirmed");
  await expect(page.locator(".return-path-result")).toContainText("Unverified build");
});

test("zero and missing counts render truthfully", async ({ page }) => {
  await stubSolve(page, {
    ...referenceResult,
    consistent_counts: [62, 0],
    turns: 2,
    rows: [
      { guess: "tares", marks: "byyyb" },
      { guess: "crane", marks: "ggggg" },
    ],
  });
  await showResult(page);
  const rows = page.locator(".return-path-row");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(1)).toContainText("0 possible words");
});

test("missing count reports unavailable, not zero", async ({ page }) => {
  await stubSolve(page, {
    ...referenceResult,
    consistent_counts: [62],
    turns: 2,
    rows: [
      { guess: "tares", marks: "byyyb" },
      { guess: "crane", marks: "ggggg" },
    ],
  });
  await showResult(page);
  const rows = page.locator(".return-path-row");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(1)).toContainText("Count unavailable");
  await expect(rows.nth(1)).not.toContainText("0 possible");
});

test("exhibition and mismatched identity never claim certified", async ({
  page,
}) => {
  await stubSolve(page, {
    ...referenceResult,
    certification: "EXHIBITION",
    identity: { proof_status: "UNVERIFIED" },
  });
  await showResult(page);
  await expect(page.locator(".return-path-certification")).toContainText(
    "Exhibition",
  );
  await expect(page.locator(".return-path-certification")).not.toContainText(
    "Certified",
  );
});

test("certified result with unverified identity fails closed", async ({
  page,
}) => {
  await stubSolve(page, {
    ...referenceResult,
    certification: "CERTIFIED",
    identity: { proof_status: "UNVERIFIED" },
  });
  await showResult(page);
  await expect(page.locator(".return-path-certification")).toContainText(
    "Unverified build",
  );
});

test("result view passes axe and selection survives about roundtrip", async ({
  page,
}) => {
  await showResult(page);
  const results = await new AxeBuilder({ page }).analyze();
  const severe = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  expect(severe.map((v) => v.id)).toEqual([]);
  const turn = page.getByRole("button", {
    name: "Inspect guess 2, DECAL",
    exact: true,
  });
  await turn.click();
  await expect(turn).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "About", exact: true }).click();
  await expect(page.locator(".about-page")).toBeVisible();
  await page.getByRole("button", { name: "← Play", exact: true }).click();
  await expect(page.locator(".return-path-row")).toHaveCount(4);
  await expect(
    page.getByRole("button", { name: "Inspect guess 2, DECAL", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
});

test("resize re-measures the trace without overflow", async ({ page }) => {
  await page.setViewportSize({ width: 1586, height: 992 });
  await showResult(page);
  const spine = page.locator(".return-path-result .trace-spine");
  const before = await spine.getAttribute("d");
  expect(before).toMatch(/^M [\d.]+ [\d.]+/);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".return-path-row")).toHaveCount(4);
  await page.waitForTimeout(150);
  const after = await spine.getAttribute("d");
  expect(after).toMatch(/^M [\d.]+ [\d.]+/);
  expect(after).not.toBe(before);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  expect(overflow).toBeLessThanOrEqual(390);
});

test("staged reveal shows only the revealed prefix", async ({ page }) => {
  // pauseAt freezes fake time; expect() polls would otherwise advance it.
  await page.clock.pauseAt(Date.now());
  await page.goto("/");
  await page.getByRole("textbox", { name: "Your word", exact: true }).fill("crane");
  await page.getByRole("button", { name: "Solve my word", exact: true }).click();
  const result = page.locator(".return-path-result");
  await expect(result.getByRole("heading", { name: "Following the guesses." }))
    .toBeVisible({ timeout: 15_000 });
  // Atomic snapshots only — locator polls advance fake time mid-reveal.
  const snap = () =>
    page.evaluate(() => ({
      rows: document.querySelectorAll(".return-path-row").length,
      text: document.querySelector(".return-path-result")?.textContent ?? "",
      footer: !!document.querySelector(".return-path-footer"),
    }));
  const before = await snap();
  expect(before.rows).toBe(0);
  expect(before.footer).toBe(false);
  expect(before.text).toContain("5,022");
  expect(before.text).not.toContain("62 possible words");
  await page.clock.runFor(500);
  const mid = await snap();
  expect(mid.rows).toBe(1);
  expect(mid.text).toContain("62 possible words");
  expect(mid.text).not.toContain("6 possible words");
  expect(mid.text).not.toContain("Confirmed");
  expect(mid.footer).toBe(false);
  await page.clock.runFor(5000);
  const after = await snap();
  expect(after.rows).toBe(4);
  expect(after.footer).toBe(true);
  expect(after.text).toContain("Solved in 4");
  await expect(result.getByRole("heading", { name: "A little less unknown." }))
    .toBeVisible();
});

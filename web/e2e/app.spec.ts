import { expect, test } from "@playwright/test";
import { AxeBuilder } from "@axe-core/playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const SHOTS = path.resolve("../reports/web/screens");
const REPORTS = path.resolve("../reports/web");
const FORBIDDEN =
  /candidate|posterior|entropy|oracle|remaining|partition|hidden/i;

type Proof = {
  label: string;
  claim?: {
    objectives: Record<
      string,
      { losses: number; score7_total: number; worst_score: number }
    >;
  };
  certificates?: Record<string, { winner: string }>;
  model?: { parameter_count: number; weights_bytes: number };
  firewall?: { checks_passed: number };
};

function forbiddenKeys(value: unknown, trail: string[] = []): string[] {
  if (Array.isArray(value))
    return value.flatMap((item, i) => forbiddenKeys(item, [...trail, `${i}`]));
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>).flatMap(
      ([key, inner]) => [
        ...(FORBIDDEN.test(key) ? [[...trail, key].join(".")] : []),
        ...forbiddenKeys(inner, [...trail, key]),
      ],
    );
  }
  return [];
}

async function championOpener(request: {
  get: (url: string) => Promise<{ json: () => Promise<unknown> }>;
}): Promise<string> {
  const proof = (await (await request.get("/model/proof.json")).json()) as Proof;
  return proof.certificates?.normal.winner.toUpperCase() ?? "TARES";
}



async function openAssist(page: import("@playwright/test").Page) {
  await page.goto("/");
  await page.evaluate(() => (window as any).__wordle);
  await page.getByRole("tab", { name: "Assist", exact: true }).click();
}

test("browser metadata", async ({ browser }) => {
  mkdirSync(REPORTS, { recursive: true });
  writeFileSync(
    path.join(REPORTS, "browser.json"),
    JSON.stringify({
      name: "chromium",
      version: browser.version(),
      playwright: "1.55.1",
    }) + "\n",
  );
});

test("desktop assist renders the trace, suggestion, passes axe", async ({
  page,
  request,
}) => {
  const proof = (await (await request.get("/model/proof.json")).json()) as Proof;
  await page.setViewportSize({ width: 1280, height: 800 });
  await openAssist(page);
  await expect(
    page.getByText("Deep Green", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".return-path-stage")).toBeVisible();
  await expect(page.locator(".return-path-origin")).toContainText(
    (proof.claim?.tier ?? 0).toLocaleString("en-US"),
  );
  const pending = page.locator(".return-path-row.is-pending");
  await expect(pending).toHaveCount(1);
  const firstRow = pending.locator(".tile");
  const championFirst =
    proof.certificates?.normal.winner.toUpperCase().split("") ?? [];
  expect(championFirst).toHaveLength(5);
  await expect(firstRow).toHaveText(championFirst);
  await expect(pending).toContainText("Suggested");
  await expect(page.locator(".trace-pending")).toHaveCount(1);
  await expect(page.locator(".return-path-footer")).toContainText("Turn 1 of 6");
  await expect(page.locator(".return-path-footer")).toContainText(
    "Exhibition · Frozen OpenWordle-v1",
  );
  const results = await new AxeBuilder({ page }).analyze();
  const severe = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  expect(
    severe.map((v) => v.id),
    "serious/critical axe violations",
  ).toEqual([]);
  mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({
    path: path.join(SHOTS, "desktop-assist.png"),
    fullPage: true,
  });
});

test("keyboard-only assist cycles marks and submits", async ({
  page,
  request,
}) => {
  const opener = await championOpener(request);
  await openAssist(page);
  await expect(page.locator(".tile").first()).toHaveText(opener[0]);
  for (let attempt = 0; attempt < 12; attempt += 1) {
    const focused = await page.evaluate(() =>
      document.activeElement?.classList.contains("tile"),
    );
    if (focused) break;
    await page.keyboard.press("Tab");
  }
  await expect
    .poll(() =>
      page.evaluate(() =>
        document.activeElement?.classList.contains("tile"),
      ),
    )
    .toBe(true);
  await page.keyboard.press("Space");
  await expect(
    page.locator(".board-row").first().locator(".tile").first(),
  ).toHaveClass(/mark-y/);
  for (let col = 0; col < 4; col += 1) {
    await page.keyboard.press("Tab");
    await page.keyboard.press("Enter");
  }
  const rowTiles = page.locator(".board-row").first().locator(".tile");
  await expect(rowTiles).toHaveCount(5);
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const onSubmit = await page.evaluate(
      () => document.activeElement?.textContent === "SUBMIT FEEDBACK",
    );
    if (onSubmit) break;
    await page.keyboard.press("Tab");
  }
  await page.keyboard.press("Enter");
  await expect(page.getByText("Turn 2 of 6", { exact: true })).toBeVisible();
  const live = page.locator("[aria-live='polite']");
  await expect(live).toHaveCount(1);
  const firstTile = page.locator(".board-row").first().locator(".tile").first();
  await expect(firstTile).toHaveClass(/mark-y/);
});

test("solve runs the policy against a typed word", async ({ page }) => {
  const apiBodies: unknown[] = [];
  const origins = new Set<string>();
  page.on("response", (response) => {
    origins.add(new URL(response.url()).origin);
    if (response.url().includes("/api/")) {
      void response
        .json()
        .then((body) => apiBodies.push(body))
        .catch(() => undefined);
    }
  });
  await page.goto("/");
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("crane");
  const letters = page.locator(".composer-letter");
  await expect(letters).toHaveText(["C", "R", "A", "N", "E"]);
  await word.press("Enter");
  const firstRow = page.locator(".board-row").first().locator(".tile");
  await expect(firstRow.first()).toHaveClass(/mark-/, { timeout: 10_000 });
  await page.waitForTimeout(50);
  for (const origin of origins) {
    expect(origin).toBe("http://127.0.0.1:3101");
  }
  for (const body of apiBodies) {
    expect(forbiddenKeys(body)).toEqual([]);
  }
});

test("mobile solve fits viewport with reachable controls", async ({
  page,
  request,
}) => {
  const proof = (await (await request.get("/model/proof.json")).json()) as Proof;
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/");
  await page.evaluate(() => (window as any).__wordle);
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  expect(scrollWidth).toBeLessThanOrEqual(375);
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await expect(word).toBeVisible();
  // The entry re-renders when the engine finishes loading; poll so a
  // remount between resolve and measure can't flake the bounds check.
  await expect
    .poll(() =>
      word.evaluate((el: HTMLElement) => {
        const box = el.getBoundingClientRect();
        return box.width > 0 && box.x >= 0 && box.x + box.width <= 375;
      }),
    )
    .toBe(true);
  const targets = page.locator("button:visible");
  const count = await targets.count();
  expect(count).toBeGreaterThan(0);
  for (let i = 0; i < count; i += 1) {
    const box = await targets.nth(i).boundingBox();
    expect(box!.height).toBeGreaterThanOrEqual(44);
  }
  mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({
    path: path.join(SHOTS, "mobile-solve.png"),
    fullPage: true,
  });
});

test("reduced motion removes tile transforms", async ({
  browser,
  request,
}) => {
  const opener = await championOpener(request);
  const context = await browser.newContext({
    baseURL: "http://127.0.0.1:3101",
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  await openAssist(page);
  await expect(page.locator(".tile").first()).toHaveText(opener[0]);
  const submit = page.getByRole("button", { name: "SUBMIT FEEDBACK" });
  await submit.click();
  await expect(page.getByText("Turn 2 of 6", { exact: true })).toBeVisible();
  const tile = page.locator(".board-row").first().locator(".tile").first();
  await expect(tile).toHaveClass(/reveal|filled/);
  const animation = await tile.evaluate(
    (node) => getComputedStyle(node).animationName,
  );
  const duration = await tile.evaluate(
    (node) => getComputedStyle(node).animationDuration,
  );
  expect(
    animation === "none" ||
      (animation.includes("opacity") && parseFloat(duration) <= 0.2),
  ).toBe(true);
  const transform = await tile.evaluate(
    (node) => getComputedStyle(node).transform,
  );
  expect(["none", "matrix(1, 0, 0, 1, 0, 0)"]).toContain(transform);
  await page.getByRole("tab", { name: "Challenge", exact: true }).click();
  const underline = page.locator(".composer-underline").first();
  await expect(underline).toBeVisible();
  const transition = await underline.evaluate(
    (node) => getComputedStyle(node).transitionDuration,
  );
  for (const part of transition.split(",")) {
    expect(parseFloat(part)).toBeLessThanOrEqual(0.2);
  }
  await context.close();
});

test("200 percent zoom keeps layout reachable without overflow", async ({
  page,
  request,
}) => {
  const proof = (await (await request.get("/model/proof.json")).json()) as Proof;
  await page.setViewportSize({ width: 640, height: 400 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "What should we solve?" }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  expect(scrollWidth).toBeLessThanOrEqual(640);
  await page.evaluate(() => (window as any).__wordle);
  await page.getByRole("button", { name: "About", exact: true }).click();
  const aboutPage = page.locator(".about-page");
  await expect(aboutPage).toContainText(proof.label, { timeout: 15_000 });
});

test("unverified proof shows only Unverified build", async ({ page }) => {
  await page.route("**/model/proof.json", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        schema: "wordle.public-proof/v1",
        status: "UNVERIFIED",
        label: "Unverified build",
      }),
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "About", exact: true }).click();
  const aboutPage = page.locator(".about-page");
  await expect(aboutPage).toContainText("Unverified build");
  await expect(aboutPage).not.toContainText("Certified");
});

test("engine error surfaces visible and live-region message", async ({
  page,
  request,
}) => {
  const opener = await championOpener(request);
  await openAssist(page);
  await page.evaluate(() =>
    (window as any).__wordle.then((engine: any) => {
      engine.assistFeedback = () => {
        throw new Error("internal error");
      };
    }),
  );
  await expect(page.locator(".tile").first()).toHaveText(opener[0]);
  await page.getByRole("button", { name: "SUBMIT FEEDBACK" }).click();
  await expect(page.locator(".error")).toContainText("internal error");
  await expect(page.locator("[aria-live='polite']")).toContainText(
    "internal error",
  );
  await expect(
    page.getByRole("button", { name: "SUBMIT FEEDBACK" }),
  ).toBeEnabled();
});

test("solve reaches a certified result and resets", async ({ page }) => {
  await page.goto("/");
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("crane");
  await word.press("Enter");
  await expect(page.locator(".result")).toBeVisible({ timeout: 15_000 });
  const result = await page.locator(".result").textContent();
  expect(result).toMatch(/Solved in [1-6]/);
  await expect(
    page.locator(".return-path-certification"),
  ).toContainText(/certified|exhibition|unverified/i);
  const solvedRow = page
    .locator(".return-path-row")
    .last()
    .locator(".tile.mark-g");
  await expect(solvedRow.first()).toBeVisible();
  const counts = page.locator(".return-path-count-value");
  expect(await counts.count()).toBeGreaterThanOrEqual(2);
  await expect(counts.last()).toHaveText("Confirmed");
  await page.getByRole("button", { name: "ANOTHER WORD" }).click();
  await expect(page.locator(".result")).toHaveCount(0);
  await expect(word).toHaveValue("");
  await expect(page.locator(".composer-underline")).toHaveCount(5);
});

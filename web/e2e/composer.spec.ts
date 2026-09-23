import { expect, test } from "@playwright/test";
import { AxeBuilder } from "@axe-core/playwright";
import { mkdirSync } from "node:fs";
import path from "node:path";

test("challenge opens without creating an assist session", async ({ page }) => {
  const apiRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/api/")) apiRequests.push(request.url());
  });
  await page.goto("/");
  await expect(page.getByRole("tab", { name: "Challenge", exact: true }))
    .toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("heading", { name: "What should we solve?" }))
    .toBeVisible();
  await expect(page.getByRole("textbox", { name: "Your word", exact: true }))
    .toHaveValue("");
  await expect(page.getByRole("button", { name: "Solve my word", exact: true }))
    .toBeVisible();
  expect(apiRequests).toEqual([]);
  await page.getByRole("tab", { name: "Assist", exact: true }).click();
  await expect(page.locator(".board-row").first().locator(".tile").first())
    .toHaveText(/[A-Z]/, { timeout: 15_000 });
  expect(apiRequests).toEqual([]);
});

test("native editing and mode changes preserve the draft", async ({ page }) => {
  await page.goto("/");
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("CRANE");
  await expect(word).toHaveValue("crane");
  await word.press("Home");
  await word.press("Delete");
  await expect(word).toHaveValue("rane");
  await word.press("c");
  await expect(word).toHaveValue("crane");
  await page.getByRole("radio", { name: "Hard", exact: true }).check();
  await expect(word).toHaveValue("crane");
  await page.getByRole("tab", { name: "Assist", exact: true }).click();
  await page.getByRole("tab", { name: "Challenge", exact: true }).click();
  await expect(word).toHaveValue("crane");
});

test("short input reports an error without solving", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() =>
    (window as any).__wordle.then((engine: any) => {
      (window as any).__solveCalls = 0;
      engine.solve = () => {
        (window as any).__solveCalls += 1;
        throw new Error("unreachable");
      };
    }),
  );
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("cat");
  await word.press("Enter");
  await expect(word).toHaveAttribute("aria-invalid", "true");
  await expect(word).toBeFocused();
  await expect(page.locator("[aria-live='polite']"))
    .toContainText("Enter a five-letter word.");
  expect(await page.evaluate(() => (window as any).__solveCalls)).toBe(0);
});

test("unknown word preserves input and recovers", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => (window as any).__wordle);
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("zzzzz");
  await word.press("Enter");
  await expect(word).toHaveValue("zzzzz");
  await expect(page.locator("[aria-live='polite']"))
    .toContainText("This word isn’t in the accepted list.");
  await expect(page.getByRole("button", { name: "Solve my word", exact: true }))
    .toBeEnabled();
  await word.fill("crane");
  await expect(word).not.toHaveAttribute("aria-invalid", "true");
});

test("capture approved word composer reference state", async ({ page }) => {
  await page.setViewportSize({ width: 1586, height: 992 });
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await page.goto("/");
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("crane");
  await word.focus();
  await word.press("End");
  await expect(word).toHaveValue("crane");
  await expect(word).toBeFocused();
  await page.evaluate(() => document.fonts.ready);
  const directory = path.resolve("../reports/web/screens");
  mkdirSync(directory, { recursive: true });
  await page.screenshot({
    path: path.join(directory, "word-composer-desktop-light.png"),
    fullPage: true,
    animations: "disabled",
  });
});

test("pending solve locks controls and runs once", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() =>
    (window as any).__wordle.then((engine: any) => {
      const original = engine.solve;
      (window as any).__solveCalls = 0;
      engine.solve = (mode: string, word: string) => {
        (window as any).__solveCalls += 1;
        return original(mode, word);
      };
    }),
  );
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("crane");
  await word.press("Enter");
  // The composer unmounts the moment solve() returns; during the staged
  // reveal state.busy keeps tabs and mode controls locked.
  await expect(
    page.getByRole("tab", { name: "Assist", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("radio", { name: "Hard", exact: true }),
  ).toBeDisabled();
  await page.keyboard.press("Enter");
  await expect(page.locator(".result")).toBeVisible({ timeout: 15_000 });
  expect(await page.evaluate(() => (window as any).__solveCalls)).toBe(1);
});

test("hard mode passes exactly mode and lowercase word", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() =>
    (window as any).__wordle.then((engine: any) => {
      const original = engine.solve;
      (window as any).__solveArgs = null;
      engine.solve = (mode: string, word: string) => {
        (window as any).__solveArgs = [mode, word];
        return original(mode, word);
      };
    }),
  );
  await page.getByRole("radio", { name: "Hard", exact: true }).check();
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("CRANE");
  await word.press("Enter");
  await expect(page.locator(".result")).toBeVisible({ timeout: 15_000 });
  expect(await page.evaluate(() => (window as any).__solveArgs)).toEqual([
    "hard",
    "crane",
  ]);
});

test("engine failures keep the draft and recover", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() =>
    (window as any).__wordle.then((engine: any) => {
      (window as any).__realSolve = engine.solve;
      engine.solve = () => {
        throw new Error("internal error");
      };
    }),
  );
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  const submit = page.getByRole("button", {
    name: "Solve my word",
    exact: true,
  });
  await word.fill("crane");
  await word.press("Enter");
  await expect(page.locator("[aria-live='polite']")).toContainText(
    "We couldn’t solve that word. Try again.",
  );
  await expect(word).toHaveValue("crane");
  await expect(word).not.toHaveAttribute("aria-invalid", "true");
  await expect(submit).toBeEnabled();
  await page.evaluate(() =>
    (window as any).__wordle.then((engine: any) => {
      engine.solve = (window as any).__realSolve;
    }),
  );
  await word.press("Enter");
  await expect(page.locator(".result")).toBeVisible({ timeout: 15_000 });
});

test("slow proof does not block entry or steal the draft", async ({
  page,
}) => {
  let release = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/model/proof.json", async (route) => {
    await gate;
    await route.continue();
  });
  await page.goto("/");
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("crane");
  await word.press("Home");
  await word.press("Shift+ArrowRight");
  await word.press("Shift+ArrowRight");
  await word.press("z");
  await expect(word).toHaveValue("zane");
  release();
  await page.waitForTimeout(300);
  await expect(word).toHaveValue("zane");
});

test("paste normalizes mixed text and replaces selections", async ({
  browser,
}) => {
  const context = await browser.newContext({
    baseURL: "http://127.0.0.1:3101",
    permissions: ["clipboard-read", "clipboard-write"],
  });
  const page = await context.newPage();
  await page.goto("/");
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.focus();
  await page.evaluate(() => navigator.clipboard.writeText(" CR!A N3E "));
  await page.keyboard.press("ControlOrMeta+V");
  await expect(word).toHaveValue("crane");
  await word.press("Home");
  await word.press("Shift+ArrowRight");
  await page.evaluate(() => navigator.clipboard.writeText("B"));
  await page.keyboard.press("ControlOrMeta+V");
  await expect(word).toHaveValue("brane");
  await page.evaluate(() =>
    navigator.clipboard.writeText("supercalifragilistic"),
  );
  await page.keyboard.press("ControlOrMeta+A");
  await page.keyboard.press("ControlOrMeta+V");
  await expect(word).toHaveValue("super");
  await context.close();
});

test("theme follows system until an explicit choice is persisted", async ({
  browser,
}) => {
  const dark = await browser.newContext({
    baseURL: "http://127.0.0.1:3101",
    colorScheme: "dark",
  });
  const page = await dark.newPage();
  await page.goto("/");
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.dataset.theme),
    )
    .toBe("dark");
  const toggle = page.getByRole("button", { name: "Switch to light mode" });
  await toggle.click();
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.dataset.theme),
    )
    .toBe("light");
  await page.reload();
  await expect
    .poll(
      () => page.evaluate(() => document.documentElement.dataset.theme),
      { timeout: 15_000 },
    )
    .toBe("light");
  await dark.close();

  const light = await browser.newContext({
    baseURL: "http://127.0.0.1:3101",
    colorScheme: "light",
  });
  const page2 = await light.newPage();
  await page2.goto("/");
  await expect
    .poll(() =>
      page2.evaluate(() => document.documentElement.dataset.theme),
    )
    .toBe("light");
  await light.close();
});

test("about page opens, shows proof and sankey, and returns", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "About", exact: true }).click();
  const aboutPage = page.locator(".about-page");
  await expect(
    aboutPage.getByRole("heading", { name: "Deep Green" }),
  ).toBeVisible();
  await expect(aboutPage.locator(".stat")).toHaveCount(4);
  await expect(aboutPage.locator(".sankey")).toBeVisible({ timeout: 10_000 });
  await aboutPage.getByRole("button", { name: "BACK TO PLAY" }).click();
  await expect(
    page.getByRole("heading", { name: "What should we solve?" }),
  ).toBeVisible();
});

test("entry passes axe in light, dark, dialog, and error states", async ({
  page,
}) => {
  const severe = async () => {
    const results = await new AxeBuilder({ page }).analyze();
    return results.violations
      .filter((v) => v.impact === "serious" || v.impact === "critical")
      .map((v) => v.id);
  };
  await page.goto("/");
  expect(await severe(), "light entry").toEqual([]);
  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.dataset.theme),
    )
    .toBe("dark");
  await page.waitForTimeout(250);
  expect(await severe(), "dark entry").toEqual([]);
  await page.getByRole("button", { name: "About", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Deep Green" }),
  ).toBeVisible();
  await expect(page.locator(".sankey")).toBeVisible({ timeout: 10_000 });
  expect(await severe(), "about page").toEqual([]);
  await page.getByRole("button", { name: "BACK TO PLAY" }).click();
  const word = page.getByRole("textbox", { name: "Your word", exact: true });
  await word.fill("cat");
  await word.press("Enter");
  await expect(word).toHaveAttribute("aria-invalid", "true");
  expect(await severe(), "error state").toEqual([]);
});

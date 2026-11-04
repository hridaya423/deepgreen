import {
  createEngine,
  EngineError,
  fetchBundle,
  fetchProof,
  loadEngine,
  modelVersion,
  type AssistState,
  type Engine,
  type EngineParts,
  type Mode,
  type ProofReceipt,
  type Row,
  type SolveResult,
} from "./engine.js";
import { observeResultTrace } from "./result-trace.js";

type Mark = "b" | "y" | "g";
type Tab = "assist" | "solve";
type Theme = "light" | "dark";

interface AppState {
  tab: Tab;
  mode: Mode;
  assist: AssistState | null;
  solve: SolveResult | null;
  solveDraft: string;
  solveShown: number;
  selectedSolveTurn: number | null;
  selectedAssistTurn: number | null;
  marks: Mark[];
  busy: boolean;
  revealRow: number | null;
  error: string;
  invalid: boolean;
  proof: ProofReceipt | null;
  aboutSolve: SolveResult | null;
  aboutLoading: boolean;
}

const state: AppState = {
  tab: "solve",
  mode: "normal",
  assist: null,
  solve: null,
  solveDraft: "",
  solveShown: 0,
  selectedSolveTurn: null,
  selectedAssistTurn: null,
  marks: ["b", "b", "b", "b", "b"],
  busy: false,
  revealRow: null,
  error: "",
  invalid: false,
  proof: null,
  aboutSolve: null,
  aboutLoading: false,
};

const root = document.getElementById("app") as HTMLElement;
if (!root) throw new Error("missing #app");

const liveRegion = document.createElement("p");
liveRegion.className = "sr-only";
liveRegion.setAttribute("aria-live", "polite");
document.body.appendChild(liveRegion);

function announce(text: string): void {
  liveRegion.textContent = "";
  if (text) {
    requestAnimationFrame(() => {
      liveRegion.textContent = text;
    });
  }
}

// Engine init is heavy (~2.5G MACs of word-key precompute); run it in a
// worker so the page never blocks. Parts come back as transferred typed
// arrays and assemble into the same synchronous Engine in ~5ms.
function engineFromWorker(base: string, version: string): Promise<Engine> {
  return new Promise((resolve, reject) => {
    const worker = new Worker("./engine.worker.js", { type: "module" });
    worker.onmessage = (event: MessageEvent) => {
      worker.terminate();
      if (event.data?.ok) {
        resolve(createEngine(event.data.parts as EngineParts));
      } else {
        reject(
          new EngineError(
            event.data?.message ?? "model load failed",
            "load-failed",
          ),
        );
      }
    };
    worker.onerror = () => {
      worker.terminate();
      reject(new Error("engine worker failed"));
    };
    worker.postMessage({ base, version });
  });
}

async function startEngine(base: string): Promise<Engine> {
  const version = modelVersion();
  try {
    return await engineFromWorker(base, version);
  } catch (error) {
    // A real load failure reported by the worker is final — only transport
    // failures (no Worker/module-worker support) fall back to inline init.
    if (error instanceof EngineError) throw error;
    return loadEngine(await fetchBundle(base, version));
  }
}

const engineReady: Promise<Engine> = (async () => {
  let engine: Engine;
  try {
    engine = await startEngine("./model/");
  } catch {
    state.proof = {
      schema: "wordle.public-proof/v1",
      status: "UNVERIFIED",
      label: "Unverified build",
    };
    render();
    throw new EngineError("the model could not be loaded", "load-failed");
  }
  try {
    engine.attachProof(await fetchProof("./model/"));
  } catch {
    /* fail closed: stays UNVERIFIED */
  }
  state.proof = engine.proof;
  render();
  return engine;
})();
engineReady.catch(() => undefined);
(window as unknown as { __wordle: Promise<Engine> }).__wordle = engineReady;

async function startAssist(): Promise<void> {
  state.error = "";
  try {
    const engine = await engineReady;
    state.assist = engine.createAssist(state.mode);
    state.marks = ["b", "b", "b", "b", "b"];
    state.selectedAssistTurn = null;
  } catch (error) {
    state.error = error instanceof Error ? error.message : "request failed";
    announce(state.error);
  }
  render();
}

async function submitFeedback(): Promise<void> {
  const session = state.assist;
  if (!session || session.terminal || state.busy) return;
  state.busy = true;
  try {
    const engine = await engineReady;
    state.assist = engine.assistFeedback(session.session_id, {
      expected_turn: session.turn,
      pending_guess: session.pending_guess,
      marks: state.marks.join(""),
    });
    state.revealRow = state.assist.rows.length - 1;
    state.marks = ["b", "b", "b", "b", "b"];
    state.selectedAssistTurn = null;
    state.error = "";
  } catch (error) {
    state.error = error instanceof Error ? error.message : "request failed";
    announce(state.error);
  } finally {
    state.busy = false;
    render();
  }
}

const HELPER_TEXT = "Choose from 5,022 accepted words.";

async function submitSolve(): Promise<void> {
  if (state.busy) return;
  const word = state.solveDraft;
  if (!/^[a-z]{5}$/.test(word)) {
    state.error = "Enter a five-letter word.";
    state.invalid = true;
    announce(state.error);
    render();
    focusWordInput();
    return;
  }
  state.busy = true;
  state.error = "";
  state.invalid = false;
  render();
  try {
    const engine = await engineReady;
    state.solve = engine.solve(state.mode, word);
    state.solveShown = 0;
    state.selectedSolveTurn = null;
    state.solveDraft = "";
    render();
    revealSolveRow();
  } catch (error) {
    state.invalid =
      error instanceof EngineError && error.code === "unknown-word";
    state.error = state.invalid
      ? "This word isn’t in the accepted list. Try another five-letter word."
      : "We couldn’t solve that word. Try again.";
    announce(state.error);
    state.solveDraft = word;
    state.solve = null;
    state.busy = false;
    render();
    focusWordInput();
  }
}

function focusWordInput(): void {
  root.querySelector<HTMLInputElement>("#challenge-word")?.focus();
}

function revealSolveRow(): void {
  const result = state.solve;
  if (!result) return;
  window.setTimeout(() => {
    state.solveShown += 1;
    state.revealRow = state.solveShown - 1;
    if (state.solveShown >= result.rows.length) state.busy = false;
    render();
    if (state.solveShown < result.rows.length) revealSolveRow();
  }, 460);
}

function resetSolve(): void {
  state.solve = null;
  state.solveDraft = "";
  state.solveShown = 0;
  state.selectedSolveTurn = null;
  state.error = "";
  state.invalid = false;
  render();
  focusWordInput();
}

function el(
  tag: string,
  className = "",
  text = "",
): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function icon(paths: string, size = 20): SVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = paths;
  return svg;
}

const sunIcon = () =>
  icon(
    '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"/>',
  );
const moonIcon = () =>
  icon('<path d="M20.4 13.2A8.5 8.5 0 1 1 10.8 3.6a7 7 0 0 0 9.6 9.6Z"/>');
const returnIcon = () =>
  icon('<path d="M9.5 10.5 5 15l4.5 4.5"/><path d="M5 15h9.5a4.5 4.5 0 0 0 0-9h-2"/>');

function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function syncThemeToggle(button: HTMLButtonElement): void {
  const toDark = currentTheme() !== "dark";
  button.setAttribute(
    "aria-label",
    toDark ? "Switch to dark mode" : "Switch to light mode",
  );
  button.textContent = "";
  button.appendChild(toDark ? sunIcon() : moonIcon());
}

function applyTheme(theme: Theme, persist: boolean): void {
  document.documentElement.dataset.theme = theme;
  if (persist) {
    try {
      localStorage.setItem("wordle-theme", theme);
    } catch {
      /* storage unavailable: session theme still applies */
    }
  }
  document
    .querySelectorAll<HTMLButtonElement>(".theme-toggle")
    .forEach(syncThemeToggle);
}

const MARK_LABEL: Record<Mark, string> = {
  b: "absent",
  y: "present",
  g: "correct",
};
const MARK_NEXT: Record<Mark, Mark> = { b: "y", y: "g", g: "b" };

function filledTile(row: Row, col: number, reveal: boolean): HTMLElement {
  const mark = (row.marks[col] ?? "b") as Mark;
  const tile = el("div", "tile filled");
  tile.classList.add(`mark-${mark}`);
  if (reveal) {
    tile.classList.add("reveal");
    tile.style.animationDelay = `${col * 85}ms`;
  }
  tile.textContent = row.guess[col]?.toUpperCase() ?? "";
  tile.setAttribute("aria-label", `${row.guess[col]} ${MARK_LABEL[mark]}`);
  return tile;
}

const TABS: Array<[Tab, string]> = [
  ["solve", "Challenge"],
  ["assist", "Assist"],
];

const panelId = (tab: Tab) =>
  tab === "solve" ? "challenge-panel" : "assist-panel";

function selectTab(tab: Tab): void {
  if (state.busy || state.tab === tab) return;
  state.tab = tab;
  state.error = "";
  state.invalid = false;
  if (tab === "assist" && !state.assist) void startAssist();
  render();
}

function renderHeader(): HTMLElement {
  const head = el("header", "app-header");
  const brand = el("div", "brand");
  const mark = el("span", "brand-mark");
  mark.setAttribute("aria-hidden", "true");
  for (let i = 0; i < 4; i += 1) mark.appendChild(document.createElement("i"));
  brand.appendChild(mark);
  brand.appendChild(el("span", "brand-name", "Deep Green"));
  head.appendChild(brand);

  const nav = el("nav", "app-tabs");
  nav.setAttribute("aria-label", "Primary");
  if (location.hash === "#about") {
    const back = el("button", "header-link", "← Play") as HTMLButtonElement;
    back.type = "button";
    back.dataset.focus = "about-back";
    back.addEventListener("click", () => {
      location.hash = "";
    });
    nav.appendChild(back);
    head.appendChild(nav);
    const utils = el("div", "header-utils");
    const theme = el("button", "theme-toggle") as HTMLButtonElement;
    theme.type = "button";
    theme.dataset.focus = "theme-toggle";
    theme.addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark", true);
    });
    syncThemeToggle(theme);
    utils.append(theme);
    head.appendChild(utils);
    return head;
  }
  const list = el("div", "tab-list");
  list.setAttribute("role", "tablist");
  list.setAttribute("aria-label", "Challenge or assist");
  for (const [tab, label] of TABS) {
    const button = el("button", "tab", label) as HTMLButtonElement;
    button.type = "button";
    button.id = `tab-${tab}`;
    button.dataset.focus = `tab-${tab}`;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(state.tab === tab));
    button.setAttribute("aria-controls", panelId(tab));
    button.tabIndex = state.tab === tab ? 0 : -1;
    button.disabled = state.busy;
    if (state.tab === tab) button.classList.add("active");
    button.addEventListener("click", () => selectTab(tab));
    button.addEventListener("keydown", (event) => {
      const order = TABS.map(([id]) => id);
      const index = order.indexOf(tab);
      let next: number | null = null;
      if (event.key === "ArrowRight") next = (index + 1) % order.length;
      else if (event.key === "ArrowLeft")
        next = (index - 1 + order.length) % order.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = order.length - 1;
      if (next === null) return;
      event.preventDefault();
      const target = order[next];
      if (!target) return;
      selectTab(target);
      root.querySelector<HTMLElement>(`#tab-${target}`)?.focus();
    });
    list.appendChild(button);
  }
  nav.appendChild(list);
  head.appendChild(nav);

  const utils = el("div", "header-utils");
  const about = el("button", "header-link", "About") as HTMLButtonElement;
  about.type = "button";
  about.dataset.focus = "about-trigger";
  about.addEventListener("click", () => {
    location.hash = "about";
  });
  const theme = el("button", "theme-toggle") as HTMLButtonElement;
  theme.type = "button";
  theme.dataset.focus = "theme-toggle";
  theme.addEventListener("click", () => {
    applyTheme(currentTheme() === "dark" ? "light" : "dark", true);
  });
  syncThemeToggle(theme);
  utils.append(about, theme);
  head.appendChild(utils);
  return head;
}

function setMode(mode: Mode): void {
  if (state.busy || state.mode === mode) return;
  state.mode = mode;
  state.assist = null;
  state.solve = null;
  state.solveShown = 0;
  state.selectedSolveTurn = null;
  state.selectedAssistTurn = null;
  state.error = "";
  state.invalid = false;
  if (state.tab === "assist") void startAssist();
  render();
}

function renderDifficulty(): HTMLElement {
  const wrap = el("div", "difficulty");
  const fieldset = el("fieldset", "modes");
  fieldset.appendChild(el("legend", "sr-only", "Difficulty"));
  for (const mode of ["normal", "hard"] as Mode[]) {
    const label = el("label", "mode-option");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "difficulty";
    radio.value = mode;
    radio.checked = state.mode === mode;
    radio.disabled = state.busy;
    radio.dataset.focus = `mode-${mode}`;
    radio.addEventListener("change", () => setMode(mode));
    label.appendChild(radio);
    label.appendChild(
      el("span", "mode-label", mode === "normal" ? "Normal" : "Hard"),
    );
    fieldset.appendChild(label);
  }
  wrap.appendChild(fieldset);
  if (state.mode === "hard") {
    wrap.appendChild(
      el(
        "p",
        "mode-note",
        "Every revealed hint must be reused — greens stay locked, yellows must reappear.",
      ),
    );
  }
  return wrap;
}

function selectAssistTurn(index: number): void {
  const session = state.assist;
  if (!session || state.busy || index < 0 || index >= session.rows.length)
    return;
  state.selectedAssistTurn =
    state.selectedAssistTurn === index ? null : index;
  render();
  if (state.selectedAssistTurn !== null)
    announce(turnSummary(session, index));
}

function renderAssist(): HTMLElement {
  const section = el("section", "return-path-result");
  const session = state.assist;
  const won = session?.terminal && session.outcome === "WON";
  const lost = session?.terminal && session.outcome === "LOST";

  const intro = el("div", "return-path-intro");
  const introText = el("div");
  introText.appendChild(
    el(
      "h1",
      "return-path-title",
      !session
        ? "Watch it narrow."
        : won
          ? "A little less unknown."
          : lost
            ? "The trace ends here."
            : "Following your board.",
    ),
  );
  introText.appendChild(
    el(
      "p",
      "return-path-subtitle",
      !session
        ? "Play your real game; record the colors here. Exhibition."
        : won
          ? `Solved in ${session.rows.length} — exhibition assist.`
          : lost
            ? "Six turns spent — exhibition assist."
            : `Turn ${session.turn} of 6 — tap a tile to cycle the color your game showed.`,
    ),
  );
  intro.append(introText, renderDifficulty());
  section.appendChild(intro);

  if (!session) {
    const start = el(
      "button",
      "primary",
      "START ASSIST",
    ) as HTMLButtonElement;
    start.dataset.focus = "assist-start";
    start.disabled = state.busy;
    start.addEventListener("click", () => void startAssist());
    section.appendChild(start);
    return section;
  }

  const total = session.rows.length;
  const stage = el("div", "return-path-stage");
  stage.style.setProperty(
    "--rp-rows",
    String(total + (session.pending_guess ? 1 : 0)),
  );
  const origin = el("p", "return-path-origin");
  origin.appendChild(
    el(
      "span",
      "",
      `${session.vocabulary_size.toLocaleString("en-US")} to start`,
    ),
  );
  const originDot = el("span", "return-path-origin-dot");
  originDot.setAttribute("data-trace-origin", "");
  originDot.setAttribute("aria-hidden", "true");
  origin.appendChild(originDot);
  stage.appendChild(origin);

  const list = el("ol", "return-path-rows");
  const selected =
    state.selectedAssistTurn !== null && state.selectedAssistTurn < total
      ? state.selectedAssistTurn
      : null;
  session.rows.forEach((row, i) => {
    const li = el("li", "return-path-row");
    const last = i === total - 1;
    if (last && session.terminal)
      li.classList.add(won ? "is-final" : "is-terminal");

    const turn = el(
      "button",
      "return-path-turn",
      String(i + 1),
    ) as HTMLButtonElement;
    turn.type = "button";
    turn.dataset.focus = `assist-turn-${i}`;
    turn.setAttribute(
      "aria-label",
      `Inspect guess ${i + 1}, ${row.guess.toUpperCase()}`,
    );
    turn.setAttribute("aria-pressed", String(i === selected));
    if (i === selected) turn.classList.add("selected");
    turn.disabled = state.busy;
    turn.addEventListener("click", () => selectAssistTurn(i));
    li.appendChild(turn);

    const tiles = el("div", "board-row return-path-word");
    tiles.setAttribute("role", "img");
    tiles.setAttribute(
      "aria-label",
      `${row.guess.toUpperCase()} — ${[...row.marks]
        .map((m) => MARK_LABEL[m as Mark] ?? m)
        .join(", ")}`,
    );
    for (let col = 0; col < 5; col += 1)
      tiles.appendChild(filledTile(row, col, i === state.revealRow));
    li.appendChild(tiles);

    const leader = el("span", "return-path-leader-anchor");
    leader.setAttribute("aria-hidden", "true");
    li.appendChild(leader);

    const count = session.consistent_counts[i];
    const countCell = el("div", "return-path-count");
    const node = el("span", "return-path-node");
    node.setAttribute("data-trace-node", "");
    node.setAttribute("aria-hidden", "true");
    countCell.appendChild(node);
    const text = el("p", "return-path-count-text");
    if (last && session.terminal) {
      text.appendChild(
        el("span", "return-path-count-value", won ? "Confirmed" : "Not solved"),
      );
      text.append(" ");
      text.appendChild(
        el("span", "return-path-count-unit", countPhrase(count)),
      );
    } else if (count === undefined) {
      text.appendChild(
        el("span", "return-path-count-value", "Count unavailable"),
      );
    } else {
      text.appendChild(
        el("span", "return-path-count-value", count.toLocaleString("en-US")),
      );
      text.append(" ");
      text.appendChild(
        el(
          "span",
          "return-path-count-unit",
          count === 1 ? "possible word" : "possible words",
        ),
      );
    }
    countCell.appendChild(text);
    li.appendChild(countCell);
    list.appendChild(li);
  });

  if (session.pending_guess) {
    const guess = session.pending_guess;
    const li = el("li", "return-path-row is-pending");
    const turn = el(
      "span",
      "return-path-turn",
      String(session.turn),
    );
    turn.setAttribute("aria-hidden", "true");
    li.appendChild(turn);

    const tiles = el("div", "board-row return-path-word");
    for (let col = 0; col < 5; col += 1) {
      const mark = state.marks[col] ?? "b";
      const tile = el("div", "tile filled", guess[col]!.toUpperCase());
      tile.classList.add(`mark-${mark}`);
      tile.setAttribute("role", "button");
      tile.setAttribute("tabindex", "0");
      tile.dataset.focus = `assist-tile-${col}`;
      tile.setAttribute(
        "aria-label",
        `Letter ${guess[col]} is ${MARK_LABEL[mark]}; activate to cycle`,
      );
      const cycle = () => {
        state.marks[col] = MARK_NEXT[mark];
        render();
      };
      tile.addEventListener("click", cycle);
      tile.addEventListener("keydown", (event) => {
        const key = (event as KeyboardEvent).key;
        if (key === "Enter" || key === " ") {
          event.preventDefault();
          cycle();
        }
      });
      tiles.appendChild(tile);
    }
    li.appendChild(tiles);

    const leader = el("span", "return-path-leader-anchor");
    leader.setAttribute("aria-hidden", "true");
    li.appendChild(leader);

    const countCell = el("div", "return-path-count");
    const node = el("span", "return-path-node");
    node.setAttribute("data-trace-node", "");
    node.setAttribute("aria-hidden", "true");
    countCell.appendChild(node);
    const text = el("p", "return-path-count-text");
    text.appendChild(el("span", "return-path-count-value", "Suggested"));
    text.append(" ");
    text.appendChild(
      el("span", "return-path-count-unit", "mark the colors"),
    );
    countCell.appendChild(text);
    li.appendChild(countCell);
    list.appendChild(li);
  }
  stage.appendChild(list);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.classList.add("return-path-lines");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  stage.appendChild(svg);
  section.appendChild(stage);

  const sel = el("p", "return-path-selection");
  if (selected !== null) {
    sel.textContent = turnSummary(session, selected);
  } else if (!session.terminal) {
    sel.classList.add(session.out_of_scope ? "scope-warning" : "scope-hint");
    sel.textContent = session.out_of_scope
      ? "No word in the certified vocabulary matches this board — check the colors you entered, or the answer may be outside the corpus."
      : session.consistent_count <= session.consistent_preview.length
        ? `matches: ${session.consistent_preview.join(", ")}`
        : `${session.consistent_count.toLocaleString("en-US")} certified words still match`;
  }
  section.appendChild(sel);

  const footer = el("footer", "return-path-footer");
  const status = el("div");
  status.appendChild(
    el(
      "p",
      "result",
      session.terminal
        ? won
          ? `Solved in ${total}`
          : "Not solved"
        : `Turn ${session.turn} of 6`,
    ),
  );
  status.appendChild(
    el("p", "return-path-certification", certificationLabel(session)),
  );
  const actions = el("div", "return-path-actions");
  if (session.pending_guess) {
    const submit = el(
      "button",
      "primary",
      "SUBMIT FEEDBACK",
    ) as HTMLButtonElement;
    submit.dataset.focus = "assist-submit";
    submit.disabled = state.busy;
    submit.addEventListener("click", () => void submitFeedback());
    actions.appendChild(submit);
  }
  const restart = el(
    "button",
    session.pending_guess ? "return-path-proof" : "primary",
    "Start over",
  ) as HTMLButtonElement;
  restart.disabled = state.busy;
  restart.addEventListener("click", () => void startAssist());
  actions.appendChild(restart);
  footer.append(status, actions);
  section.appendChild(footer);
  return section;
}

function normalizeWordEdit(value: string, start: number, end: number) {
  const letters = (text: string) => text.toLowerCase().replace(/[^a-z]/g, "");
  const word = letters(value).slice(0, 5);
  return {
    word,
    start: Math.min(letters(value.slice(0, start)).length, word.length),
    end: Math.min(letters(value.slice(0, end)).length, word.length),
  };
}

let composerSync: (() => void) | null = null;

function renderComposer(): HTMLElement {
  const form = el("form", "word-composer") as HTMLFormElement;
  form.setAttribute("aria-busy", String(state.busy));

  const intro = el("div", "composer-intro");
  const introText = el("div");
  introText.appendChild(
    el("h1", "composer-heading", "What should we solve?"),
  );
  introText.appendChild(
    el("p", "composer-desc", "Enter a five-letter word."),
  );
  intro.appendChild(introText);
  intro.appendChild(renderDifficulty());
  form.appendChild(intro);

  const field = el("div", "composer-field");
  const label = el("label", "composer-label", "Your word");
  label.setAttribute("for", "challenge-word");
  field.appendChild(label);

  const editor = el("div", "composer-editor");
  const input = el("input", "composer-input") as HTMLInputElement;
  input.id = "challenge-word";
  input.name = "word";
  input.type = "text";
  input.autocomplete = "off";
  input.setAttribute("autocapitalize", "characters");
  input.setAttribute("inputmode", "text");
  input.spellcheck = false;
  input.setAttribute("aria-describedby", "composer-message");
  input.dataset.focus = "solve-word";
  input.value = state.solveDraft;
  input.readOnly = state.busy;
  if (state.invalid) input.setAttribute("aria-invalid", "true");

  const visual = el("div", "composer-visual");
  visual.setAttribute("aria-hidden", "true");
  const columns = el("div", "composer-columns");
  const colEls: HTMLElement[] = [];
  const letterEls: HTMLElement[] = [];
  const underEls: HTMLElement[] = [];
  for (let i = 0; i < 5; i += 1) {
    const col = el("div", "composer-column");
    const letter = el("span", "composer-letter");
    const under = el("span", "composer-underline");
    col.append(letter, under);
    columns.appendChild(col);
    colEls.push(col);
    letterEls.push(letter);
    underEls.push(under);
  }
  const selectionEl = el("div", "composer-selection");
  selectionEl.hidden = true;
  const caretEl = el("div", "composer-caret");
  caretEl.hidden = true;
  visual.append(columns, selectionEl, caretEl);
  editor.append(input, visual);
  field.appendChild(editor);
  form.appendChild(field);

  const actions = el("div", "composer-actions");
  const message = el(
    "p",
    "composer-message",
    state.error || HELPER_TEXT,
  );
  message.id = "composer-message";
  if (state.error) message.classList.add("is-error");
  actions.appendChild(message);
  const submit = el("button", "composer-submit") as HTMLButtonElement;
  submit.type = "submit";
  submit.disabled = state.busy;
  submit.dataset.focus = "solve-submit";
  submit.append(
    el("span", "", state.busy ? "Solving…" : "Solve my word"),
    returnIcon(),
  );
  actions.appendChild(submit);
  form.appendChild(actions);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void submitSolve();
  });

  let composing = false;
  let selAnchor = 0;

  const letterFont = () =>
    parseFloat(getComputedStyle(letterEls[0] ?? input).fontSize) || 16;

  const caretX = (pos: number): number => {
    const len = input.value.length;
    const last = letterEls[4];
    if (len === 5 && pos === 5 && last) {
      return last.offsetLeft + last.offsetWidth + 4;
    }
    const letter = letterEls[pos];
    if (pos < len && letter) return letter.offsetLeft - 4;
    const col = colEls[Math.min(pos, 4)];
    if (!col) return 0;
    return col.offsetLeft + (col.offsetWidth - letterFont() * 0.6) / 2;
  };

  const sync = () => {
    const value = input.value;
    state.solveDraft = value;
    const focused = document.activeElement === input;
    const start = input.selectionStart ?? 0;
    const end = input.selectionEnd ?? start;
    const collapsed = start === end;
    for (let i = 0; i < 5; i += 1) {
      const ch = value[i];
      const letter = letterEls[i];
      const under = underEls[i];
      if (!letter || !under) continue;
      letter.textContent = ch ? ch.toUpperCase() : "";
      under.classList.toggle(
        "active",
        focused && collapsed && Math.min(start, 4) === i,
      );
    }
    if (focused && collapsed) {
      caretEl.hidden = false;
      caretEl.style.transform = `translateX(${caretX(start)}px)`;
    } else {
      caretEl.hidden = true;
    }
    const first = colEls[Math.min(start, 4)];
    const last = colEls[Math.min(end - 1, 4)];
    if (focused && !collapsed && first && last) {
      selectionEl.hidden = false;
      selectionEl.style.left = `${first.offsetLeft}px`;
      selectionEl.style.width = `${last.offsetLeft + last.offsetWidth - first.offsetLeft}px`;
    } else {
      selectionEl.hidden = true;
    }
  };
  composerSync = sync;

  const afterEdit = () => {
    if (state.error) {
      state.error = "";
      state.invalid = false;
      input.removeAttribute("aria-invalid");
      message.textContent = HELPER_TEXT;
      message.classList.remove("is-error");
    }
    sync();
  };

  const normalizeNow = () => {
    const start = input.selectionStart ?? 0;
    const end = input.selectionEnd ?? start;
    const next = normalizeWordEdit(input.value, start, end);
    if (next.word !== input.value) {
      input.value = next.word;
      input.setSelectionRange(next.start, next.end);
    }
    afterEdit();
  };

  input.addEventListener("input", () => {
    if (!composing) normalizeNow();
  });
  input.addEventListener("compositionstart", () => {
    composing = true;
  });
  input.addEventListener("compositionend", () => {
    composing = false;
    normalizeNow();
  });
  input.addEventListener("paste", (event) => {
    event.preventDefault();
    const text = event.clipboardData?.getData("text") ?? "";
    const start = input.selectionStart ?? 0;
    const end = input.selectionEnd ?? start;
    const merged =
      input.value.slice(0, start) + text + input.value.slice(end);
    const next = normalizeWordEdit(merged, start, start + text.length);
    input.value = next.word;
    input.setSelectionRange(next.end, next.end);
    afterEdit();
  });
  input.addEventListener("dblclick", () => {
    input.setSelectionRange(0, input.value.length);
    sync();
  });

  const hitIndex = (clientX: number): number => {
    const len = input.value.length;
    for (let i = 0; i < 5; i += 1) {
      const col = colEls[i];
      if (!col) continue;
      const rect = col.getBoundingClientRect();
      if (clientX < rect.left + rect.width / 2) return Math.min(i, len);
    }
    return len;
  };

  input.addEventListener("pointerdown", (event) => {
    if (input.readOnly) return;
    event.preventDefault();
    input.focus({ preventScroll: true });
    const index = hitIndex(event.clientX);
    if (event.shiftKey) {
      input.setSelectionRange(selAnchor, index);
    } else {
      selAnchor = index;
      input.setSelectionRange(index, index);
    }
    sync();
    const move = (ev: PointerEvent) => {
      input.setSelectionRange(selAnchor, hitIndex(ev.clientX));
      sync();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });

  for (const type of ["keyup", "select", "click", "focus", "blur"]) {
    input.addEventListener(type, sync);
  }
  sync();
  return form;
}

const TURN_WORDS = ["One", "Two", "Three", "Four", "Five", "Six"];

function countPhrase(count: number | undefined): string {
  if (count === undefined) return "Count unavailable";
  return `${count.toLocaleString("en-US")} possible word${count === 1 ? "" : "s"}`;
}

function certificationLabel(result: {
  certification: string;
  identity: SolveResult["identity"];
}): string {
  if (result.certification === "CERTIFIED") {
    if (result.identity?.proof_status === "VERIFIED")
      return "Certified · Frozen OpenWordle-v1";
    console.warn("CERTIFIED solve with unverified identity", result.identity);
    return "Unverified build";
  }
  if (result.certification === "EXHIBITION")
    return "Exhibition · Frozen OpenWordle-v1";
  return "Unverified build";
}

function turnSummary(
  result: Pick<SolveResult, "rows" | "consistent_counts" | "vocabulary_size">,
  index: number,
): string {
  const row = result.rows[index];
  const before =
    index === 0
      ? result.vocabulary_size
      : result.consistent_counts[index - 1];
  const after = result.consistent_counts[index];
  const fmt = (n: number | undefined) =>
    n === undefined ? "?" : n.toLocaleString("en-US");
  return `Guess ${index + 1} · ${row?.guess.toUpperCase() ?? "?"} · ${fmt(before)} → ${fmt(after)} possible word${after === 1 ? "" : "s"}`;
}

function selectSolveTurn(index: number): void {
  const result = state.solve;
  if (!result || state.busy || index < 0 || index >= result.rows.length) return;
  state.selectedSolveTurn = index;
  render();
  announce(turnSummary(result, index));
}

function renderSolveResult(): HTMLElement {
  const section = el("section", "return-path-result");
  const result = state.solve;
  if (!result) return section;
  const total = result.rows.length;
  const complete = state.solveShown >= total;
  const lastRow = result.rows[total - 1];
  const solved =
    complete &&
    result.outcome === "SOLVED" &&
    lastRow?.marks === "ggggg";
  const word = result.word.toUpperCase();

  const intro = el("div", "return-path-intro");
  const introText = el("div");
  introText.appendChild(
    el(
      "h1",
      "return-path-title",
      !complete
        ? "Following the guesses."
        : solved
          ? "A little less unknown."
          : "The trace ends here.",
    ),
  );
  introText.appendChild(
    el(
      "p",
      "return-path-subtitle",
      !complete
        ? `Solving ${word}…`
        : solved
          ? `${TURN_WORDS[total - 1] ?? String(total)} guess${total === 1 ? "" : "es"} to ${word}.`
          : `${word} wasn’t solved in ${total} guesses.`,
    ),
  );
  intro.append(introText, renderDifficulty());
  section.appendChild(intro);

  const stage = el("div", "return-path-stage");
  stage.style.setProperty("--rp-rows", String(total));
  const origin = el("p", "return-path-origin");
  origin.appendChild(
    el("span", "", `${result.vocabulary_size.toLocaleString("en-US")} to start`),
  );
  const originDot = el("span", "return-path-origin-dot");
  originDot.setAttribute("data-trace-origin", "");
  originDot.setAttribute("aria-hidden", "true");
  origin.appendChild(originDot);
  stage.appendChild(origin);

  const list = el("ol", "return-path-rows");
  const explicit =
    state.selectedSolveTurn !== null && state.selectedSolveTurn < total
      ? state.selectedSolveTurn
      : null;
  const selected = explicit ?? (complete ? total - 1 : state.solveShown - 1);
  result.rows.slice(0, state.solveShown).forEach((row, i) => {
    const li = el("li", "return-path-row");
    const last = i === total - 1;
    if (last && complete) li.classList.add(solved ? "is-final" : "is-terminal");

    const turn = el(
      "button",
      "return-path-turn",
      String(i + 1),
    ) as HTMLButtonElement;
    turn.type = "button";
    turn.dataset.focus = `trace-turn-${i}`;
    turn.setAttribute(
      "aria-label",
      `Inspect guess ${i + 1}, ${row.guess.toUpperCase()}`,
    );
    turn.setAttribute("aria-pressed", String(i === selected));
    if (i === selected) turn.classList.add("selected");
    turn.disabled = state.busy;
    turn.addEventListener("click", () => selectSolveTurn(i));
    li.appendChild(turn);

    const tiles = el("div", "board-row return-path-word");
    tiles.setAttribute("role", "img");
    tiles.setAttribute(
      "aria-label",
      `${row.guess.toUpperCase()} — ${[...row.marks]
        .map((m) => MARK_LABEL[m as Mark] ?? m)
        .join(", ")}`,
    );
    for (let col = 0; col < 5; col += 1)
      tiles.appendChild(filledTile(row, col, i === state.revealRow));
    li.appendChild(tiles);

    const leader = el("span", "return-path-leader-anchor");
    leader.setAttribute("aria-hidden", "true");
    li.appendChild(leader);

    const count = result.consistent_counts[i];
    const countCell = el("div", "return-path-count");
    const node = el("span", "return-path-node");
    node.setAttribute("data-trace-node", "");
    node.setAttribute("aria-hidden", "true");
    countCell.appendChild(node);
    const text = el("p", "return-path-count-text");
    if (last && complete) {
      text.appendChild(
        el(
          "span",
          "return-path-count-value",
          solved ? "Confirmed" : "Not solved",
        ),
      );
      text.append(" ");
      text.appendChild(
        el("span", "return-path-count-unit", countPhrase(count)),
      );
    } else if (count === undefined) {
      text.appendChild(
        el("span", "return-path-count-value", "Count unavailable"),
      );
    } else {
      text.appendChild(
        el("span", "return-path-count-value", count.toLocaleString("en-US")),
      );
      text.append(" ");
      text.appendChild(
        el(
          "span",
          "return-path-count-unit",
          count === 1 ? "possible word" : "possible words",
        ),
      );
    }
    countCell.appendChild(text);
    li.appendChild(countCell);
    list.appendChild(li);
  });
  stage.appendChild(list);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.classList.add("return-path-lines");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  stage.appendChild(svg);
  section.appendChild(stage);

  const sel = el("p", "return-path-selection");
  if (complete && selected !== null && selected >= 0 && selected < total - 1)
    sel.textContent = turnSummary(result, selected);
  section.appendChild(sel);

  if (complete) {
    const footer = el("footer", "return-path-footer");
    const status = el("div");
    status.appendChild(
      el("p", "result", solved ? `Solved in ${total}` : "Not solved"),
    );
    status.appendChild(
      el("p", "return-path-certification", certificationLabel(result)),
    );
    const actions = el("div", "return-path-actions");
    const again = el(
      "button",
      "primary",
      "Another word",
    ) as HTMLButtonElement;
    again.dataset.focus = "solve-again";
    again.addEventListener("click", resetSolve);
    const proof = el(
      "a",
      "return-path-proof",
      "View proof",
    ) as HTMLAnchorElement;
    proof.href = "#about";
    actions.append(again, proof);
    footer.append(status, actions);
    section.appendChild(footer);
  }
  return section;
}

let disposeResultTrace: (() => void) | null = null;

let aboutStages = 0;

function sankeyDiagram(
  stages: Array<{ word: string; count: number; solved?: boolean }>,
  fresh = false,
): SVGElement {
  const n = stages.length;
  const vocab = Math.max(1, stages[0]?.count ?? 1);
  const topPad = 30;
  const usable = 140;
  const bottomPad = 38;
  const colW = 8;
  const padX = 40;
  const W = 380;
  const H = topPad + usable + bottomPad;
  const gap = n > 1 ? Math.min(120, (W - 2 * padX - n * colW) / (n - 1)) : 0;
  const total = n * colW + (n - 1) * gap;
  const x0 = (W - total) / 2;
  const colX = (i: number) => x0 + i * (colW + gap);
  const scale = usable / vocab;
  const band = (c: number) => (c <= 0 ? 0 : Math.max(2, c * scale));
  const topY = (c: number) => topPad + (usable - band(c)) / 2;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.classList.add("sankey");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `Words still possible after each guess: ${stages.map((s) => s.count.toLocaleString("en-US")).join(", then ")}`,
  );

  const field = document.createElementNS(SVG_NS, "rect");
  field.setAttribute("x", String(x0));
  field.setAttribute("y", String(topPad));
  field.setAttribute("width", String(total));
  field.setAttribute("height", String(usable));
  field.setAttribute("rx", "2");
  field.classList.add("sk-field");
  svg.appendChild(field);

  for (let i = 0; i < n - 1; i += 1) {
    const cur = stages[i];
    const nxt = stages[i + 1];
    if (!cur || !nxt || band(nxt.count) <= 0) continue;
    const x1 = colX(i) + colW;
    const x2 = colX(i + 1);
    const mid = (x1 + x2) / 2;
    const t1 = topY(cur.count);
    const t2 = topY(nxt.count);
    const b1 = t1 + band(cur.count);
    const b2 = t2 + band(nxt.count);
    const link = document.createElementNS(SVG_NS, "path");
    link.setAttribute(
      "d",
      `M ${x1} ${t1} C ${mid} ${t1} ${mid} ${t2} ${x2} ${t2}` +
        ` L ${x2} ${b2} C ${mid} ${b2} ${mid} ${b1} ${x1} ${b1} Z`,
    );
    link.classList.add("sk-alive");
    if (fresh && i === n - 2) link.classList.add("sk-new");
    svg.appendChild(link);
    const flow = document.createElementNS(SVG_NS, "path");
    flow.setAttribute(
      "d",
      `M ${x1} ${(t1 + b1) / 2} C ${mid} ${(t1 + b1) / 2} ${mid} ${(t2 + b2) / 2} ${x2} ${(t2 + b2) / 2}`,
    );
    flow.classList.add("sk-flow");
    svg.appendChild(flow);
  }

  stages.forEach((stage, i) => {
    const cx = colX(i) + colW / 2;
    const word = document.createElementNS(SVG_NS, "text");
    word.setAttribute("x", String(cx));
    word.setAttribute("y", String(topPad - 10));
    word.setAttribute("text-anchor", "middle");
    word.classList.add("sk-word");
    word.textContent = stage.word;
    svg.appendChild(word);

    if (band(stage.count) > 0) {
      const node = document.createElementNS(SVG_NS, "rect");
      node.setAttribute("x", String(colX(i)));
      node.setAttribute("y", String(topY(stage.count)));
      node.setAttribute("width", String(colW));
      node.setAttribute("height", String(band(stage.count)));
      node.classList.add("sk-alive");
      if (fresh && i === n - 1 && i > 0) node.classList.add("sk-new");
      const tip = document.createElementNS(SVG_NS, "title");
      tip.textContent = `${stage.word}: ${stage.count.toLocaleString("en-US")} possible`;
      node.appendChild(tip);
      svg.appendChild(node);
    }

    const count = document.createElementNS(SVG_NS, "text");
    count.setAttribute("x", String(cx));
    count.setAttribute("y", String(topPad + usable + 18));
    count.setAttribute("text-anchor", "middle");
    count.classList.add("sk-count");
    if (stage.solved) count.classList.add("ok");
    count.textContent = stage.solved
      ? "✓"
      : stage.count.toLocaleString("en-US");
    svg.appendChild(count);

    const prev = stages[i - 1];
    const ruled = i > 0 && prev ? prev.count - stage.count : 0;
    if (ruled > 0) {
      const delta = document.createElementNS(SVG_NS, "text");
      delta.setAttribute("x", String(cx));
      delta.setAttribute("y", String(topPad + usable + 31));
      delta.setAttribute("text-anchor", "middle");
      delta.classList.add("sk-delta");
      delta.textContent = `−${ruled.toLocaleString("en-US")}`;
      svg.appendChild(delta);
    }
  });
  return svg;
}

async function loadAboutSolve(): Promise<void> {
  if (state.aboutSolve || state.aboutLoading) return;
  state.aboutLoading = true;
  try {
    const engine = await engineReady;
    state.aboutSolve = engine.solve("normal", "fjord");
  } catch {
    state.aboutSolve = null;
  } finally {
    state.aboutLoading = false;
    if (location.hash === "#about") render();
  }
}

function renderAbout(): HTMLElement {
  const page = el("div", "about-page");
  const proof = state.proof;

  const hero = el("section", "about-hero");
  hero.appendChild(el("h1", "about-title", "Deep Green"));
  hero.appendChild(
    el(
      "p",
      "about-tag",
      "A 1.55-million-parameter neural net that plays provably optimal Wordle.",
    ),
  );
  page.appendChild(hero);

  const stats = el("section", "stat-grid");
  const stat = (value: string, label: string) => {
    const cell = el("div", "stat");
    cell.appendChild(el("p", "stat-value", value));
    cell.appendChild(el("p", "stat-label", label));
    stats.appendChild(cell);
  };
  const tier = proof?.claim?.tier;
  const normal = proof?.claim?.objectives.normal;
  stat(tier ? tier.toLocaleString("en-US") : "—", "words in the field");
  stat(
    normal ? `${normal.losses}` : "—",
    "losses, either mode",
  );
  stat(
    normal ? `${normal.worst_score}` : "—",
    "turns, worst case",
  );
  stat(
    proof?.model ? proof.model.parameter_count.toLocaleString("en-US") : "—",
    "parameters",
  );
  page.appendChild(stats);

  const think = el("section", "about-section");
  think.appendChild(el("p", "overline", "WATCH IT THINK"));
  const holder = el("div", "about-sankey");
  if (state.aboutSolve) {
    const aboutStagesCount = state.aboutSolve.rows.length + 1;
    const aboutFresh = aboutStagesCount > aboutStages;
    aboutStages = aboutStagesCount;
    holder.appendChild(
      sankeyDiagram(
        [
          { word: "ALL WORDS", count: state.aboutSolve.vocabulary_size },
          ...state.aboutSolve.rows.map((row, i) => ({
            word: row.guess.toUpperCase(),
            count: state.aboutSolve!.consistent_counts[i] ?? 0,
            solved: row.marks === "ggggg",
          })),
        ],
        aboutFresh,
      ),
    );
    holder.appendChild(
      el("p", "sankey-cap", "words that could still be the secret"),
    );
    think.appendChild(holder);
    think.appendChild(
      el(
        "p",
        "about-note",
        `A real solve — ${state.aboutSolve.word.toUpperCase()}, recomputed when this page opened.`,
      ),
    );
  } else {
    holder.appendChild(
      el("p", "about-note", "Recomputing a solve…"),
    );
    think.appendChild(holder);
  }
  page.appendChild(think);

  const how = el("section", "about-section");
  how.appendChild(el("p", "overline", "HOW IT WORKS"));
  const steps = el("ol", "about-steps");
  for (const step of [
    "Reads only the public board — past guesses and their colors.",
    "Scores every word in the 5,022-word vocabulary.",
    "Plays the argmax. No search, no tables — six megabytes of distilled strategy.",
  ]) {
    steps.appendChild(el("li", "", step));
  }
  how.appendChild(steps);
  page.appendChild(how);

  const verified = el("section", "about-section");
  verified.appendChild(el("p", "overline", "VERIFIED"));
  verified.appendChild(
    el(
      "p",
      "about-copy",
      "All 10,346 reachable positions were replayed and matched the certified optimal solver decision-for-decision, in normal and hard mode. Certificates, census and this interface regenerate in a clean worktree; the release is hash-sealed.",
    ),
  );
  verified.appendChild(
    el(
      "p",
      `about-status ${proof?.status === "VERIFIED" ? "ok" : "warn"}`,
      proof?.status === "VERIFIED"
        ? `${proof.label} — verified`
        : "Unverified build",
    ),
  );
  page.appendChild(verified);

  const corpus = el("section", "about-section");
  corpus.appendChild(el("p", "overline", "THE WORDS"));
  corpus.appendChild(
    el(
      "p",
      "about-copy",
      "OpenWordle-v1 — 5,022 five-letter words rebuilt from open-licensed sources (SCOWL and WordNet). Not the NYT list; an open reconstruction of the playable vocabulary.",
    ),
  );
  page.appendChild(corpus);

  const back = el("button", "primary", "BACK TO PLAY") as HTMLButtonElement;
  back.dataset.focus = "about-back";
  back.addEventListener("click", () => {
    location.hash = "";
  });
  page.appendChild(back);
  return page;
}

function render(): void {
  disposeResultTrace?.();
  disposeResultTrace = null;
  const active = document.activeElement as HTMLElement | null;
  const focusKey = active?.dataset?.focus;
  const selection =
    active instanceof HTMLInputElement && focusKey === "solve-word"
      ? {
          start: active.selectionStart ?? 0,
          end: active.selectionEnd ?? 0,
          dir: active.selectionDirection ?? "none",
        }
      : null;
  root.textContent = "";
  const onAbout = location.hash === "#about";
  const entry = !onAbout && state.tab === "solve" && !state.solve;
  const resultView = !onAbout && state.tab === "solve" && state.solve !== null;
  const shell = el(
    "div",
    entry ? "shell shell-entry" : resultView ? "shell shell-result" : "shell",
  );
  shell.appendChild(renderHeader());
  const main = el("main");
  if (onAbout) {
    main.appendChild(renderAbout());
  } else if (entry) {
    const panel = el("div", "entry-panel");
    panel.id = panelId("solve");
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", "tab-solve");
    panel.appendChild(renderComposer());
    main.appendChild(panel);
  } else if (resultView) {
    const panel = el("div", "result-panel");
    panel.id = panelId("solve");
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", "tab-solve");
    panel.appendChild(renderSolveResult());
    if (state.error) panel.appendChild(el("p", "error", state.error));
    main.appendChild(panel);
  } else {
    const panel = el("div", "result-panel");
    panel.id = panelId(state.tab);
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${state.tab}`);
    panel.appendChild(renderAssist());
    if (state.error) panel.appendChild(el("p", "error", state.error));
    main.appendChild(panel);
  }
  shell.appendChild(main);
  const credit = el("footer", "site-credit");
  credit.append("Built by ");
  const creditLink = el("a", "", "Hridya Agrawal") as HTMLAnchorElement;
  creditLink.href = "https://www.hridya.tech/projects/deepgreen";
  credit.appendChild(creditLink);
  shell.appendChild(credit);
  root.appendChild(shell);
  const traceStage = root.querySelector<HTMLElement>(".return-path-stage");
  if (traceStage) {
    disposeResultTrace = observeResultTrace(
      traceStage,
      state.tab === "assist"
        ? (state.revealRow ?? -1)
        : state.busy
          ? state.solveShown - 1
          : -1,
    );
  }
  state.revealRow = null;
  if (focusKey) {
    const restored = root.querySelector<HTMLElement>(
      `[data-focus="${focusKey}"]`,
    );
    restored?.focus({ preventScroll: true });
    if (selection && restored instanceof HTMLInputElement) {
      restored.setSelectionRange(
        selection.start,
        selection.end,
        selection.dir as "forward" | "backward" | "none",
      );
      composerSync?.();
    }
  }
}

window.addEventListener("resize", () => composerSync?.());
void document.fonts?.ready.then(() => composerSync?.());

{
  const scheme = matchMedia("(prefers-color-scheme: dark)");
  scheme.addEventListener("change", () => {
    let stored: string | null = null;
    try {
      stored = localStorage.getItem("wordle-theme");
    } catch {
      stored = null;
    }
    if (!stored) applyTheme(scheme.matches ? "dark" : "light", false);
  });
}

window.addEventListener("hashchange", () => {
  if (location.hash === "#about") void loadAboutSolve();
  render();
});

render();
if (location.hash === "#about") void loadAboutSolve();

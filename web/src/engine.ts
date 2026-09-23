/*
 * In-browser port of the certified policy runtime.
 *
 * Mirrors src/deep_green/{rules,numeric_policy,webapp}.py so the whole
 * app deploys as static files: weights.safetensors is fetched once, scored
 * with Float32Array math, and the proof receipt is re-verified against the
 * downloaded bytes with WebCrypto. Any integrity mismatch fails closed to
 * UNVERIFIED, same as load_public_proof.
 */

export type Mode = "normal" | "hard";

export interface Row {
  guess: string;
  marks: string;
}

export interface Identity {
  proof_status: string;
  weights_sha256?: string;
  manifest_sha256?: string;
}

export interface AssistState {
  session_id: string;
  mode: Mode;
  turn: number;
  rows: Row[];
  pending_guess: string | null;
  consistent_count: number;
  consistent_counts: number[];
  vocabulary_size: number;
  consistent_preview: string[];
  out_of_scope: boolean;
  terminal: boolean;
  outcome: string | null;
  certification: string;
  identity: Identity;
}

export interface SolveResult {
  mode: Mode;
  word: string;
  rows: Row[];
  turns: number;
  consistent_counts: number[];
  vocabulary_size: number;
  outcome: string;
  certification: string;
  identity: Identity;
}

export interface ProofReceipt {
  schema: string;
  status: string;
  label: string;
  claim?: {
    tier?: number;
    objectives: Record<
      string,
      { losses: number; score7_total: number; worst_score: number }
    >;
  };
  model?: {
    parameter_count: number;
    weights_bytes: number;
    max_abs_logit_delta: number;
  };
  release?: { weights_sha256: string; manifest_sha256: string };
  certificates?: Record<string, { answers_sha256: string }>;
}

export class EngineError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message);
    this.name = "EngineError";
  }
}

export interface EngineBundle {
  weights: ArrayBuffer;
  words: ArrayBuffer;
  manifest: ArrayBuffer;
}

// Content version injected by build.mjs; lets model files carry immutable
// cache headers while deploys still roll cleanly.
export function modelVersion(): string {
  try {
    return (
      document
        .querySelector('meta[name="model-version"]')
        ?.getAttribute("content") ?? ""
    );
  } catch {
    return "";
  }
}

const versioned = (name: string, version: string): string =>
  version ? `${name}?v=${version}` : name;

export async function fetchBundle(
  base = "./model/",
  version = modelVersion(),
): Promise<EngineBundle> {
  const get = async (name: string): Promise<ArrayBuffer> => {
    const response = await fetch(base + versioned(name, version));
    if (!response.ok) throw new EngineError(`missing ${name}`, "load-failed");
    return response.arrayBuffer();
  };
  const [weights, words, manifest] = await Promise.all([
    get("weights.safetensors"),
    get("accepted-guesses.txt"),
    get("payload-manifest.json"),
  ]);
  return { weights, words, manifest };
}

export async function fetchProof(
  base = "./model/",
  version = modelVersion(),
): Promise<ArrayBuffer> {
  const response = await fetch(base + versioned("proof.json", version));
  if (!response.ok) throw new EngineError("missing proof.json", "load-failed");
  return response.arrayBuffer();
}

const WORD_LEN = 5;
const MAX_ROWS = 6;
const LETTERS = 26;
const MARKS = 3;
const POSITIONS = 5;
const MARK_CODE: Record<string, number> = { b: 0, y: 1, g: 2 };
const UNVERIFIED: ProofReceipt = {
  schema: "wordle.public-proof/v1",
  status: "UNVERIFIED",
  label: "Unverified build",
};

function expectedShapes(width: number): Record<string, number[]> {
  const d = width;
  return {
    "shared.letters": [LETTERS, d],
    "shared.marks": [MARKS, d],
    "shared.positions": [POSITIONS, d],
    "row.letter_mix": [d, d],
    "row.mark_mix": [d, d],
    "row.position_mix": [d, d],
    "row.sum_mix": [d, d],
    "row.max_mix": [d, d],
    "row.letter_bias": [d],
    "row.mark_bias": [d],
    "row.position_bias": [d],
    "board.project": [d + 3, d],
    "board.w1": [d, d],
    "board.w2": [d, d],
    "board.b1": [d],
    "board.b2": [d],
    "board.turn_gate": [d],
    "word.letter_mix": [d, d],
    "word.position_mix": [d, d],
    "word.sum_mix": [d, d],
    "word.max_mix": [d, d],
    "word.key_mix": [d, d],
    "word.out_mix": [d, d],
    "word.letter_bias": [d],
    "score.bilinear": [d, d],
    "score.bias": [d],
  };
}

function parseSafetensors(
  buf: ArrayBuffer,
): { tensors: Map<string, Float32Array>; width: number } {
  const view = new DataView(buf);
  const headerLen = Number(view.getBigUint64(0, true));
  if (headerLen <= 0 || 8 + headerLen > buf.byteLength) {
    throw new EngineError("bad safetensors header", "load-failed");
  }
  const raw = JSON.parse(
    new TextDecoder().decode(buf.slice(8, 8 + headerLen)),
  ) as Record<string, unknown>;
  const meta = raw["__metadata__"] as Record<string, string> | undefined;
  const width = Number(meta?.["width"]);
  const header = raw as Record<
    string,
    { dtype?: string; shape?: number[]; data_offsets?: number[] }
  >;
  if (!Number.isInteger(width) || width <= 0) {
    throw new EngineError("missing model width", "load-failed");
  }
  const shapes = expectedShapes(width);
  const tensors = new Map<string, Float32Array>();
  const dataStart = 8 + headerLen;
  for (const [name, spec] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (spec.dtype !== "F32" || !spec.data_offsets || !spec.shape) {
      throw new EngineError(`tensor ${name} is not F32`, "load-failed");
    }
    const want = shapes[name];
    if (!want || want.length !== spec.shape.length) {
      throw new EngineError(`tensor ${name} unknown`, "load-failed");
    }
    let count = 1;
    for (let i = 0; i < want.length; i += 1) {
      if (want[i] !== spec.shape[i]) {
        throw new EngineError(`tensor ${name} shape drift`, "load-failed");
      }
      count *= want[i]!;
    }
    const [start, end] = spec.data_offsets;
    if (end! - start! !== count * 4 || dataStart + end! > buf.byteLength) {
      throw new EngineError(`tensor ${name} extent drift`, "load-failed");
    }
    const data = new Float32Array(buf.slice(dataStart + start!, dataStart + end!));
    if (data.length !== count || !data.every(Number.isFinite)) {
      throw new EngineError(`tensor ${name} nonfinite`, "load-failed");
    }
    tensors.set(name, data);
  }
  if (tensors.size !== Object.keys(shapes).length) {
    throw new EngineError("tensor set incomplete", "load-failed");
  }
  return { tensors, width };
}

export async function sha256hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
}

/* ---- rules.py ports ---- */

function scoreGuess(guess: string, answer: string): string {
  const marks = ["b", "b", "b", "b", "b"];
  const remaining = new Array<number>(LETTERS).fill(0);
  for (const ch of answer) remaining[ch.charCodeAt(0) - 97]! += 1;
  for (let i = 0; i < WORD_LEN; i += 1) {
    if (answer[i] === guess[i]) {
      marks[i] = "g";
      remaining[guess.charCodeAt(i) - 97]! -= 1;
    }
  }
  for (let i = 0; i < WORD_LEN; i += 1) {
    if (marks[i] === "g") continue;
    const code = guess.charCodeAt(i) - 97;
    if (remaining[code]! > 0) {
      marks[i] = "y";
      remaining[code]! -= 1;
    }
  }
  return marks.join("");
}

function consistentAnswers(rows: Row[], answers: string[]): string[] {
  return answers.filter((answer) =>
    rows.every((row) => scoreGuess(row.guess, answer) === row.marks),
  );
}

function legalMask(mode: Mode, rows: Row[], words: string[]): boolean[] {
  if (mode === "normal") return new Array<boolean>(words.length).fill(true);
  const greens: Array<[number, number]> = [];
  const required = new Array<number>(LETTERS).fill(0);
  for (const row of rows) {
    const revealed = new Array<number>(LETTERS).fill(0);
    for (let i = 0; i < WORD_LEN; i += 1) {
      const mark = row.marks[i];
      const code = row.guess.charCodeAt(i) - 97;
      if (mark === "g") greens.push([i, code]);
      if (mark === "g" || mark === "y") revealed[code]! += 1;
    }
    for (let c = 0; c < LETTERS; c += 1) {
      required[c] = Math.max(required[c]!, revealed[c]!);
    }
  }
  return words.map((word) => {
    for (const [index, code] of greens) {
      if (word.charCodeAt(index) - 97 !== code) return false;
    }
    if (required.some((need) => need > 0)) {
      const available = new Array<number>(LETTERS).fill(0);
      for (const ch of word) available[ch.charCodeAt(0) - 97]! += 1;
      for (let c = 0; c < LETTERS; c += 1) {
        if (available[c]! < required[c]!) return false;
      }
    }
    return true;
  });
}

/* ---- numeric_policy.py port (Float32Array storage, f64 accumulate) ---- */

type Tensors = Map<string, Float32Array>;

function vecmat(v: ArrayLike<number>, m: Float32Array, width: number): Float32Array {
  const out = new Float32Array(width);
  for (let j = 0; j < v.length; j += 1) {
    const vj = v[j]!;
    if (vj === 0) continue;
    const off = j * width;
    for (let k = 0; k < width; k += 1) out[k] = out[k]! + vj * m[off + k]!;
  }
  return out;
}

function addBias(v: Float32Array, bias: Float32Array): Float32Array {
  for (let i = 0; i < v.length; i += 1) v[i] = v[i]! + bias[i]!;
  return v;
}

function relu(v: Float32Array): Float32Array {
  for (let i = 0; i < v.length; i += 1) if (v[i]! < 0) v[i] = 0;
  return v;
}

function addInto(
  dst: Float32Array,
  a: Float32Array,
  b: Float32Array,
): Float32Array {
  for (let i = 0; i < dst.length; i += 1) dst[i] = a[i]! + b[i]!;
  return dst;
}

export interface Engine {
  readonly words: string[];
  readonly proof: ProofReceipt;
  attachProof(bytes: ArrayBuffer): void;
  identity(): Identity;
  nextGuess(mode: Mode, rows: Row[]): string;
  createAssist(mode: Mode): AssistState;
  assistFeedback(
    sessionId: string,
    body: { expected_turn: number; pending_guess: string | null; marks: string },
  ): AssistState;
  solve(mode: Mode, word: string): SolveResult;
}

interface Session {
  mode: Mode;
  rows: Row[];
  pendingGuess: string;
  consistent: string[];
  consistentCounts: number[];
  terminal: boolean;
  outcome: string | null;
}

export interface EngineParts {
  tensors: Map<string, Float32Array>;
  width: number;
  words: string[];
  weightsSha: string;
  wordsSha: string;
  manifestSha: string;
  gateVec: Float32Array;
  rowLetter: Float32Array[];
  rowMark: Float32Array[];
  rowPos: Float32Array[];
  wordLetter: Float32Array[];
  wordPos: Float32Array[];
  keys: Float32Array;
  keyBias: Float32Array;
}

// Every Float32Array the worker transfers back to the main thread.
// Deduped: cached keys/keyBias are views into one shared buffer.
export function collectBuffers(parts: EngineParts): ArrayBuffer[] {
  const buffers: ArrayBuffer[] = [];
  const seen = new Set<ArrayBuffer>();
  const push = (array: Float32Array): void => {
    const buffer = array.buffer as ArrayBuffer;
    if (!seen.has(buffer)) {
      seen.add(buffer);
      buffers.push(buffer);
    }
  };
  parts.tensors.forEach(push);
  push(parts.gateVec);
  for (const list of [
    parts.rowLetter,
    parts.rowMark,
    parts.rowPos,
    parts.wordLetter,
    parts.wordPos,
  ])
    list.forEach(push);
  push(parts.keys);
  push(parts.keyBias);
  return buffers;
}

// All heavy init: safetensors parse, digest verification, embedding mixes,
// and the per-word key precompute (~2.5G MACs). Safe to run in a worker.
// `cachedKeys` (from Cache API, keyed by weightsSha) skips the key loop.
export async function prepareEngine(
  bundle: EngineBundle,
  cachedKeys?: { keys: Float32Array; keyBias: Float32Array },
): Promise<EngineParts> {
  const { tensors, width } = parseSafetensors(bundle.weights);
  const text = new TextDecoder().decode(bundle.words);
  const words = text.split("\n").filter((line) => line.length > 0);
  if (
    !words.every((word) => /^[a-z]{5}$/.test(word)) ||
    new Set(words).size !== words.length ||
    words.some((word, i) => i > 0 && word <= words[i - 1]!)
  ) {
    throw new EngineError("action vocabulary failed validation", "load-failed");
  }
  const manifest = JSON.parse(new TextDecoder().decode(bundle.manifest)) as {
    schema?: string;
    files?: Record<string, { sha256?: string }>;
    identity?: { action_space_sha256?: string };
  };
  const weightsSha = await sha256hex(bundle.weights);
  const wordsSha = await sha256hex(bundle.words);
  const manifestSha = await sha256hex(bundle.manifest);
  if (
    manifest.schema !== "wordle.release-payload/v1" ||
    manifest.files?.["weights.safetensors"]?.sha256 !== weightsSha ||
    manifest.files["accepted-guesses.txt"]?.sha256 !== wordsSha ||
    manifest.identity?.action_space_sha256 !== wordsSha
  ) {
    throw new EngineError("release bundle digest mismatch", "load-failed");
  }

  const t = (name: string): Float32Array => tensors.get(name)!;
  const gateVec = t("board.turn_gate").map((v) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v)))));

  // Embedding slices are constant: fold mix+bias once so each slot is one add+relu.
  const mixEmbed = (
    table: Float32Array,
    rows: number,
    mix: Float32Array,
    bias: Float32Array | null,
  ): Float32Array[] => {
    const out: Float32Array[] = [];
    for (let r = 0; r < rows; r += 1) {
      const slice = table.subarray(r * width, (r + 1) * width);
      const part = vecmat(slice, mix, width);
      if (bias) addBias(part, bias);
      out.push(part);
    }
    return out;
  };
  const rowLetter = mixEmbed(t("shared.letters"), LETTERS, t("row.letter_mix"), t("row.letter_bias"));
  const rowMark = mixEmbed(t("shared.marks"), MARKS, t("row.mark_mix"), t("row.mark_bias"));
  const rowPos = mixEmbed(t("shared.positions"), POSITIONS, t("row.position_mix"), t("row.position_bias"));
  const wordLetter = mixEmbed(t("shared.letters"), LETTERS, t("word.letter_mix"), t("word.letter_bias"));
  const wordPos = mixEmbed(t("shared.positions"), POSITIONS, t("word.position_mix"), null);

  // Per-word keys and biases are board-independent; compute once.
  const count = words.length;
  let keys: Float32Array;
  let keyBias: Float32Array;
  if (
    cachedKeys &&
    cachedKeys.keys.length === count * width &&
    cachedKeys.keyBias.length === count
  ) {
    keys = cachedKeys.keys;
    keyBias = cachedKeys.keyBias;
  } else {
    keys = new Float32Array(count * width);
    keyBias = new Float32Array(count);
    const perPos = new Float32Array(POSITIONS * width);
    const summed = new Float32Array(width);
    const maxed = new Float32Array(width);
    const wordBuf = new Uint8Array(WORD_LEN);
    for (let w = 0; w < count; w += 1) {
      const word = words[w]!;
      for (let i = 0; i < WORD_LEN; i += 1) wordBuf[i] = word.charCodeAt(i) - 97;
      summed.fill(0);
      maxed.fill(-Infinity);
      for (let p = 0; p < POSITIONS; p += 1) {
        const lp = wordLetter[wordBuf[p]!]!;
        const pp = wordPos[p]!;
        const slot = perPos.subarray(p * width, (p + 1) * width);
        for (let k = 0; k < width; k += 1) {
          const val = Math.max(lp[k]! + pp[k]!, 0);
          slot[k] = val;
          summed[k]! += val;
          if (val > maxed[k]!) maxed[k] = val;
        }
      }
      const pooled = vecmat(summed, t("word.sum_mix"), width);
      addInto(pooled, pooled, vecmat(maxed, t("word.max_mix"), width));
      relu(pooled);
      const key = vecmat(
        relu(vecmat(pooled, t("word.key_mix"), width)),
        t("word.out_mix"),
        width,
      );
      keys.set(key, w * width);
      let bias = 0;
      const scoreBias = t("score.bias");
      for (let k = 0; k < width; k += 1) bias += key[k]! * scoreBias[k]!;
      keyBias[w] = bias;
    }
  }
  return {
    tensors,
    width,
    words,
    weightsSha,
    wordsSha,
    manifestSha,
    gateVec,
    rowLetter,
    rowMark,
    rowPos,
    wordLetter,
    wordPos,
    keys,
    keyBias,
  };
}

// Synchronous assembly of the Engine closure — runs on whichever thread
// ends up with the parts (main thread normally, inline as fallback).
export function createEngine(parts: EngineParts): Engine {
  const {
    tensors,
    width,
    words,
    weightsSha,
    wordsSha,
    manifestSha,
    gateVec,
    rowLetter,
    rowMark,
    rowPos,
    wordLetter,
    wordPos,
    keys,
    keyBias,
  } = parts;
  const wordIndex = new Map(words.map((word, i) => [word, i]));
  const count = words.length;
  const t = (name: string): Float32Array => tensors.get(name)!;

  let proof = UNVERIFIED;
  let certifiedScope = false;
  const attachProof = (bytes: ArrayBuffer): void => {
    try {
      const receipt = JSON.parse(new TextDecoder().decode(bytes)) as ProofReceipt;
      const certs = receipt.certificates;
      const certified =
        receipt.status === "VERIFIED" &&
        receipt.schema === "wordle.public-proof/v1" &&
        receipt.release?.weights_sha256 === weightsSha &&
        receipt.release?.manifest_sha256 === manifestSha &&
        receipt.claim?.tier === words.length &&
        typeof certs === "object" &&
        certs !== null &&
        Object.keys(certs).sort().join(",") === "hard,normal" &&
        Object.values(certs).every(
          (record) => record?.answers_sha256 === wordsSha,
        );
      if (certified) {
        proof = receipt;
        certifiedScope = true;
      }
    } catch {
      /* fail closed: stays UNVERIFIED */
    }
  };

  const boardState = (mode: Mode, rows: Row[]): Float32Array => {
    const slots = Math.max(rows.length, 1);
    const summed = new Float32Array(width);
    const maxed = new Float32Array(width).fill(-Infinity);
    const perRow = new Float32Array(slots * width);
    rows.forEach((row, r) => {
      const target = perRow.subarray(r * width, (r + 1) * width);
      for (let s = 0; s < WORD_LEN; s += 1) {
        const lp = rowLetter[row.guess.charCodeAt(s) - 97]!;
        const mp = rowMark[MARK_CODE[row.marks[s]!]!]!;
        const pp = rowPos[s]!;
        for (let k = 0; k < width; k += 1) {
          target[k]! += Math.max(lp[k]! + mp[k]! + pp[k]!, 0);
        }
      }
      for (let k = 0; k < width; k += 1) {
        summed[k]! += target[k]!;
        if (target[k]! > maxed[k]!) maxed[k] = target[k]!;
      }
    });
    if (rows.length === 0) maxed.fill(0);
    const pooled = new Float32Array(width);
    const sumMix = vecmat(summed, t("row.sum_mix"), width);
    const maxMix = vecmat(maxed, t("row.max_mix"), width);
    for (let k = 0; k < width; k += 1) {
      pooled[k] = Math.max(sumMix[k]! + maxMix[k]!, 0) * gateVec[k]!;
    }
    const trunk = new Float32Array(width + 3);
    trunk.set(pooled);
    trunk[width] = rows.length + 1;
    trunk[width + 1] = mode === "normal" ? 1 : 0;
    trunk[width + 2] = mode === "hard" ? 1 : 0;
    const hidden = relu(addBias(vecmat(trunk, t("board.project"), width), t("board.b1")));
    return vecmat(relu(addBias(vecmat(hidden, t("board.w1"), width), t("board.b2"))), t("board.w2"), width);
  };

  const sessions = new Map<string, Session>();

  const identity = (): Identity => {
    const result: Identity = { proof_status: proof.status };
    if (proof.status === "VERIFIED" && proof.release) {
      result.weights_sha256 = proof.release.weights_sha256;
      result.manifest_sha256 = proof.release.manifest_sha256;
    }
    return result;
  };

  const nextGuess = (mode: Mode, rows: Row[]): string => {
    const mask = legalMask(mode, rows, words);
    const board = boardState(mode, rows);
    const q = vecmat(board, t("score.bilinear"), width);
    let best = -1;
    let bestValue = -Infinity;
    for (let w = 0; w < count; w += 1) {
      if (!mask[w]) continue;
      let logit = keyBias[w]!;
      const off = w * width;
      for (let k = 0; k < width; k += 1) logit += keys[off + k]! * q[k]!;
      if (logit > bestValue) {
        bestValue = logit;
        best = w;
      }
    }
    if (best < 0) throw new EngineError("no legal action", "empty-mask");
    return words[best]!;
  };

  const assistPayload = (id: string, session: Session): AssistState => ({
    session_id: id,
    mode: session.mode,
    turn: session.rows.length + 1,
    rows: session.rows.map((row) => ({ ...row })),
    pending_guess: session.terminal
      ? null
      : session.pendingGuess || null,
    consistent_count: session.consistent.length,
    consistent_counts: [...session.consistentCounts],
    vocabulary_size: count,
    consistent_preview: session.consistent.slice(0, 8),
    out_of_scope: session.consistent.length === 0,
    terminal: session.terminal,
    outcome: session.outcome,
    certification: "EXHIBITION",
    identity: identity(),
  });

  return {
    words,
    get proof() {
      return proof;
    },
    attachProof,
    identity,
    nextGuess,

    createAssist(mode: Mode): AssistState {
      const id = crypto.randomUUID();
      const session: Session = {
        mode,
        rows: [],
        pendingGuess: nextGuess(mode, []),
        consistent: [...words],
        consistentCounts: [],
        terminal: false,
        outcome: null,
      };
      sessions.set(id, session);
      return assistPayload(id, session);
    },

    assistFeedback(
      sessionId: string,
      body: { expected_turn: number; pending_guess: string | null; marks: string },
    ): AssistState {
      const session = sessions.get(sessionId);
      if (!session) throw new EngineError("unknown session", "unknown-session");
      if (session.terminal) {
        throw new EngineError("session is terminal", "session-closed");
      }
      if (body.expected_turn !== session.rows.length + 1) {
        throw new EngineError(
          "expected_turn does not match session",
          "stale-turn",
        );
      }
      if (body.pending_guess !== session.pendingGuess) {
        throw new EngineError(
          "pending_guess does not match the issued guess",
          "pending-mismatch",
        );
      }
      if (!/^[byg]{5}$/.test(body.marks)) {
        throw new EngineError("malformed marks", "malformed");
      }
      session.rows.push({ guess: session.pendingGuess, marks: body.marks });
      session.consistent = consistentAnswers(session.rows, session.consistent);
      session.consistentCounts.push(session.consistent.length);
      if (body.marks === "ggggg") {
        session.terminal = true;
        session.outcome = "WON";
      } else if (session.rows.length >= MAX_ROWS) {
        session.terminal = true;
        session.outcome = "LOST";
      } else if (session.consistent.length === 0) {
        session.pendingGuess = "";
      } else {
        let guess = nextGuess(session.mode, session.rows);
        const played = new Set(session.rows.map((row) => row.guess));
        if (
          played.has(guess) ||
          (session.consistent.length <= 2 && !session.consistent.includes(guess))
        ) {
          guess = session.consistent[0]!;
        }
        session.pendingGuess = guess;
      }
      return assistPayload(sessionId, session);
    },

    solve(mode: Mode, word: string): SolveResult {
      if (!wordIndex.has(word)) {
        throw new EngineError(
          "word is outside the certified vocabulary",
          "unknown-word",
        );
      }
      const rows: Row[] = [];
      const remaining: number[] = [];
      let pool: string[] = [...words];
      while (rows.length < MAX_ROWS) {
        const guess = nextGuess(mode, rows);
        const marks = scoreGuess(guess, word);
        rows.push({ guess, marks });
        pool = consistentAnswers(rows, pool);
        remaining.push(pool.length);
        if (marks === "ggggg") break;
      }
      const inTier = wordIndex.has(word);
      const certification =
        inTier && certifiedScope
          ? "CERTIFIED"
          : inTier
            ? "UNVERIFIED"
            : "EXHIBITION";
      return {
        mode,
        word,
        rows,
        turns: rows.length,
        consistent_counts: remaining,
        vocabulary_size: count,
        outcome: rows[rows.length - 1]!.marks === "ggggg" ? "SOLVED" : "UNSOLVED",
        certification,
        identity: identity(),
      };
    },
  };
}

export async function loadEngine(bundle: EngineBundle): Promise<Engine> {
  return createEngine(await prepareEngine(bundle));
}

/* Measured decorative trace for the Challenge result: leaders from each word
   to its path node, one spine from the vocabulary origin through the nodes,
   and a return segment beneath a solved word. Rebuilt from DOM anchors on
   every observe/update so geometry follows layout, not assumptions. */

const SVG_NS = "http://www.w3.org/2000/svg";

type TracePoint = { x: number; y: number };

function traceGeometry(
  origin: TracePoint,
  nodes: readonly TracePoint[],
  returnEnd: TracePoint | null,
  baselineY: number | null,
): { spine: string; returning: string | null } {
  let previous = origin;
  let spine = `M ${origin.x} ${origin.y}`;
  for (const node of nodes) {
    if (Math.abs(node.x - previous.x) < 0.5) {
      spine += ` L ${node.x} ${node.y}`;
    } else {
      const middleY = previous.y + (node.y - previous.y) / 2;
      spine += ` C ${previous.x} ${middleY} ${node.x} ${middleY} ${node.x} ${node.y}`;
    }
    previous = node;
  }
  if (nodes.length === 0 || returnEnd === null || baselineY === null) {
    return { spine, returning: null };
  }
  const bend = Math.min(28, Math.max(0, (previous.x - returnEnd.x) / 2));
  const returning =
    `M ${previous.x} ${previous.y}` +
    ` Q ${previous.x} ${baselineY} ${previous.x - bend} ${baselineY}` +
    ` L ${returnEnd.x} ${baselineY}`;
  return { spine, returning };
}

/* Returns a disposer: disconnects the ResizeObserver and cancels any queued
   frame. The caller re-creates the observer on each render, so freshIndex is
   a snapshot of which row (if any) was just revealed. */
export function observeResultTrace(
  stage: HTMLElement,
  freshIndex: number,
): () => void {
  const svg = stage.querySelector<SVGSVGElement>(".return-path-lines");
  let raf = 0;
  const update = () => {
    if (!svg || !stage.isConnected) return;
    const box = stage.getBoundingClientRect();
    const w = Math.max(0, box.width);
    const h = Math.max(0, box.height);
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("width", String(w));
    svg.setAttribute("height", String(h));
    svg.textContent = "";
    const center = (node: Element | null): TracePoint | null => {
      const r = node?.getBoundingClientRect();
      return r
        ? {
            x: r.left + r.width / 2 - box.left,
            y: r.top + r.height / 2 - box.top,
          }
        : null;
    };
    const add = (
      tag: string,
      className: string,
      attrs: Record<string, string>,
    ): void => {
      const node = document.createElementNS(SVG_NS, tag);
      node.setAttribute("class", className);
      node.setAttribute("vector-effect", "non-scaling-stroke");
      for (const [key, value] of Object.entries(attrs))
        node.setAttribute(key, value);
      svg.appendChild(node);
    };
    const origin = center(stage.querySelector("[data-trace-origin]"));
    const rows = Array.from(
      stage.querySelectorAll<HTMLElement>(".return-path-row"),
    );
    const nodes: TracePoint[] = [];
    const leaders: Array<{ d: string; fresh: boolean }> = [];
    const dots: Array<{ at: TracePoint; kind: string; fresh: boolean }> = [];
    rows.forEach((li, i) => {
      const node = center(li.querySelector("[data-trace-node]"));
      if (!node) return;
      nodes.push(node);
      const terminal = li.classList.contains("is-final")
        ? "check"
        : li.classList.contains("is-terminal")
          ? "ring"
          : li.classList.contains("is-pending")
            ? "pending"
            : "node";
      dots.push({ at: node, kind: terminal, fresh: i === freshIndex });
      if (terminal === "check") return;
      const anchor = li
        .querySelector(".return-path-leader-anchor")
        ?.getBoundingClientRect();
      const word = li
        .querySelector(".return-path-word")
        ?.getBoundingClientRect();
      let start: TracePoint | null = null;
      if (anchor && word && (anchor.width > 0 || anchor.height > 0)) {
        start = {
          x: anchor.left - box.left,
          y: word.top + word.height / 2 - box.top,
        };
      } else {
        const text = li
          .querySelector(".return-path-count-text")
          ?.getBoundingClientRect();
        if (text)
          start = {
            x: text.right - box.left,
            y: text.top + text.height / 2 - box.top,
          };
      }
      if (!start) return;
      const d =
        Math.abs(node.y - start.y) < 0.5
          ? `M ${start.x} ${start.y} L ${node.x} ${node.y}`
          : `M ${start.x} ${start.y} C ${(start.x + node.x) / 2} ${start.y} ${(start.x + node.x) / 2} ${node.y} ${node.x} ${node.y}`;
      leaders.push({ d, fresh: i === freshIndex });
    });
    let returnEnd: TracePoint | null = null;
    let baselineY: number | null = null;
    const finalWord = rows
      .find((li) => li.classList.contains("is-final"))
      ?.querySelector(".return-path-word")
      ?.getBoundingClientRect();
    if (finalWord) {
      baselineY = finalWord.bottom - box.top + 6;
      returnEnd = { x: finalWord.left - box.left - 14, y: baselineY };
    }
    for (const leader of leaders) {
      add(
        "path",
        leader.fresh ? "trace-leader trace-fade" : "trace-leader",
        { d: leader.d },
      );
    }
    if (origin) {
      const { spine, returning } = traceGeometry(
        origin,
        nodes,
        returnEnd,
        baselineY,
      );
      if (nodes.length > 0) add("path", "trace-spine", { d: spine });
      add("circle", "trace-dot", {
        cx: String(origin.x),
        cy: String(origin.y),
        r: "6",
      });
      for (const dot of dots) {
        const cx = String(dot.at.x);
        const cy = String(dot.at.y);
        const fade = dot.fresh ? " trace-fade" : "";
        if (dot.kind === "check") {
          add("circle", `trace-check${fade}`, { cx, cy, r: "21" });
          add("path", `trace-tick${fade}`, {
            d: `M ${dot.at.x - 9} ${dot.at.y + 0.5} L ${dot.at.x - 2.5} ${dot.at.y + 6.5} L ${dot.at.x + 9.5} ${dot.at.y - 7}`,
          });
        } else if (dot.kind === "ring") {
          add("circle", `trace-ring${fade}`, { cx, cy, r: "7" });
        } else if (dot.kind === "pending") {
          add("circle", `trace-pending${fade}`, { cx, cy, r: "7" });
        } else {
          add("circle", `trace-node${fade}`, { cx, cy, r: "7" });
        }
      }
      if (returning !== null && returnEnd !== null) {
        add("path", "trace-return trace-fade", { d: returning });
        add("circle", "trace-dot trace-fade", {
          cx: String(returnEnd.x),
          cy: String(returnEnd.y),
          r: "5",
        });
      }
    }
  };
  const schedule = () => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      update();
    });
  };
  schedule();
  const observer = new ResizeObserver(schedule);
  observer.observe(stage);
  void document.fonts?.ready.then(schedule);
  return () => {
    observer.disconnect();
    if (raf) cancelAnimationFrame(raf);
  };
}

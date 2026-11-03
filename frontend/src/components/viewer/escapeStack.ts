import { useEffect, useRef } from "react";

/**
 * One Escape handler for every viewer/popup: Escape closes only the
 * top-most open layer, never the panels underneath it (MainPage's global
 * Escape handler runs later and only sees Escape when no layer is open).
 */
type Layer = { close: () => void };
const stack: Layer[] = [];

function onKeyDown(e: KeyboardEvent) {
  if (e.key !== "Escape" || stack.length === 0) return;
  e.preventDefault();
  e.stopImmediatePropagation();
  stack[stack.length - 1]!.close();
}

let installed = false;
function install() {
  if (installed || typeof window === "undefined") return;
  // Window capture runs before any document/element listener.
  window.addEventListener("keydown", onKeyDown, true);
  installed = true;
}

/** Register an Escape layer while `active`; the latest `onEscape` is always used. */
export function useEscapeLayer(onEscape: () => void, active = true): void {
  const ref = useRef(onEscape);
  ref.current = onEscape;
  useEffect(() => {
    if (!active) return;
    install();
    const layer: Layer = { close: () => ref.current() };
    stack.push(layer);
    return () => {
      const i = stack.indexOf(layer);
      if (i >= 0) stack.splice(i, 1);
    };
  }, [active]);
}

/** True when a viewer layer is open (for global shortcuts that should yield). */
export const hasOpenEscapeLayer = () => stack.length > 0;

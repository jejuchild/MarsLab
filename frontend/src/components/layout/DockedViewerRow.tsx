import { useState } from "react";
import type { ReactNode } from "react";

/**
 * A docked analysis viewer (SHARAD radargram, 3D terrain) next to the
 * Inspector, so the comparison at the inspected point stays in view instead
 * of being replaced. The Inspector folds to a strip when space is short.
 */
export default function DockedViewerRow({
  viewer,
  renderInspector,
}: {
  viewer: ReactNode;
  /** Render the Inspector; its own collapse button folds it into the strip. */
  renderInspector: (onCollapse: () => void) => ReactNode;
}) {
  const [inspectorOpen, setInspectorOpen] = useState(
    () => typeof window === "undefined" || window.innerWidth >= 1600,
  );

  return (
    <div className="flex h-full">
      <div className="h-full min-w-0">{viewer}</div>
      {inspectorOpen ? (
        <div className="h-full border-l border-border-dark">{renderInspector(() => setInspectorOpen(false))}</div>
      ) : (
        <button
          type="button"
          onClick={() => setInspectorOpen(true)}
          className="flex h-full w-9 flex-col items-center gap-2 border-l border-border-dark bg-[#0a0f18] pt-3 text-slate-400 transition-colors hover:bg-[#121a28] hover:text-white"
          aria-label="Show Inspector next to the viewer"
          title="Show the Inspector (compare at this point)"
        >
          <span className="material-symbols-outlined text-[18px]">chevron_left</span>
          <span
            className="text-[10px] font-bold uppercase tracking-widest"
            style={{ writingMode: "vertical-lr", textOrientation: "mixed" }}
          >
            Inspector
          </span>
        </button>
      )}
    </div>
  );
}

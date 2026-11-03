import { useEffect, useRef, useState } from "react";
import { useEscapeLayer } from "../viewer/escapeStack";
import { INSTRUMENTS, INSTRUMENT_GROUPS } from "../../config/instrumentRegistry";
import type { InstrumentGroup, InstrumentId } from "../../config/instrumentRegistry";
import type { FootprintCount, InstrumentVisibility } from "../layerpanel/types";

export interface InstrumentBarProps {
  instrumentVisibility: InstrumentVisibility;
  onToggleInstrument: (id: InstrumentId, v: boolean) => void;
  onLoadFootprints: (instrument: string) => void;
  footprintsLoading: Record<string, boolean>;
  footprintCounts: Record<string, FootprintCount>;
  highResOnly: boolean;
  onHighResOnlyChange: (v: boolean) => void;
}

const load = (props: InstrumentBarProps, id: InstrumentId) => {
  props.onToggleInstrument(id, true);
  props.onLoadFootprints(id.toUpperCase());
};

const countOf = (c: FootprintCount | undefined) => (c ? c.count : 0);

function formatCount(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : String(n);
}

/**
 * Footprint controls on the map itself — the primary MarsLab action
 * ("show me what each instrument has here") one click away instead of
 * collapsed at the bottom of the side panel. One chip per instrument
 * (SHARAD / CRISM / HiRISE / CTX); the caret opens its variants.
 */
export default function InstrumentBar(props: InstrumentBarProps) {
  const { footprintCounts, instrumentVisibility, highResOnly, onHighResOnlyChange } = props;
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement>(null);

  // Close the variant menu on outside click / Escape (Escape is consumed so
  // it doesn't also clear the selection behind the map).
  useEffect(() => {
    if (!openGroup) return;
    const onDown = (e: PointerEvent) => {
      if (!barRef.current?.contains(e.target as Node)) setOpenGroup(null);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [openGroup]);
  useEscapeLayer(() => setOpenGroup(null), openGroup !== null);

  const loadedIds = (Object.keys(footprintCounts) as InstrumentId[]).filter((id) => footprintCounts[id]);
  const anyLoaded = loadedIds.length > 0;
  const nothingInView = anyLoaded && loadedIds.every((id) => countOf(footprintCounts[id]) === 0);

  const toggleHighResOnly = () => {
    onHighResOnlyChange(!highResOnly);
    // Re-fetch the visible loaded layers once the filter has propagated.
    setTimeout(() => {
      for (const id of loadedIds) if (instrumentVisibility[id]) props.onLoadFootprints(id.toUpperCase());
    }, 100);
  };

  return (
    <div
      ref={barRef}
      className="absolute top-3 left-3 right-16 z-30 flex flex-wrap items-start gap-1.5 pointer-events-none"
      role="toolbar"
      aria-label="Instrument footprints"
    >
      {INSTRUMENT_GROUPS.map((group) => (
        <GroupChip
          key={group.id}
          group={group}
          bar={props}
          open={openGroup === group.id}
          onOpenChange={(o) => setOpenGroup(o ? group.id : null)}
        />
      ))}

      <button
        type="button"
        onClick={toggleHighResOnly}
        aria-pressed={highResOnly}
        title="Only show products whose full-resolution data is downloaded"
        className={`pointer-events-auto flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-[11px] font-semibold shadow-lg backdrop-blur-md transition-colors ${
          highResOnly
            ? "border-purple-500/60 bg-purple-500/25 text-purple-200"
            : "border-border-dark bg-surface-dark/90 text-slate-400 hover:text-white"
        }`}
      >
        <span className="material-symbols-outlined text-[16px]">hd</span>
        Downloaded only
      </button>

      {!anyLoaded && (
        <span className="pointer-events-none self-center rounded-md bg-black/50 px-2 py-1 text-[10px] text-slate-300">
          Pick an instrument to show its footprints in this view
        </span>
      )}
      {nothingInView && (
        <span className="pointer-events-none self-center rounded-md bg-black/50 px-2 py-1 text-[10px] text-amber-200">
          No products in this view — zoom out or pan, then reload (↻)
        </span>
      )}
    </div>
  );
}

function GroupChip({
  group,
  bar,
  open,
  onOpenChange,
}: {
  group: InstrumentGroup;
  bar: InstrumentBarProps;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { footprintCounts, footprintsLoading, instrumentVisibility, onToggleInstrument } = bar;
  const ids = group.instruments;
  const primary = ids[0]!;
  const loaded = ids.filter((id) => footprintCounts[id]);
  const shown = loaded.filter((id) => instrumentVisibility[id]);
  const loading = ids.some((id) => footprintsLoading[id]);
  const on = shown.length > 0;
  const total = shown.reduce((s, id) => s + countOf(footprintCounts[id]), 0);
  const truncated = shown.some((id) => footprintCounts[id]?.truncated);
  const colour = INSTRUMENTS[(shown[0] ?? primary)].color;

  const onMain = () => {
    if (loading) return;
    if (loaded.length === 0) load(bar, primary);
    else for (const id of loaded) onToggleInstrument(id, !on);
  };

  const state = loading ? "loading" : on ? "shown" : loaded.length ? "hidden" : "not loaded";

  return (
    <div className="pointer-events-auto relative">
      <div
        className={`flex h-9 items-stretch overflow-hidden rounded-lg border shadow-lg backdrop-blur-md transition-colors ${
          on ? "border-white/25 bg-surface-dark/95" : "border-border-dark bg-surface-dark/80"
        }`}
        style={on ? { boxShadow: `inset 3px 0 0 ${colour}` } : undefined}
      >
        <button
          type="button"
          onClick={onMain}
          aria-pressed={on}
          aria-label={`${group.displayName} footprints: ${state}`}
          title={loaded.length === 0 ? `Load ${group.displayName} footprints for this view` : on ? "Hide" : "Show"}
          className={`flex items-center gap-2 pl-3 pr-2 text-[12px] font-bold tracking-wide ${
            on ? "text-white" : "text-slate-400 hover:text-white"
          }`}
        >
          {loading ? (
            <span className="material-symbols-outlined animate-spin text-[16px] text-slate-300">progress_activity</span>
          ) : (
            <span
              className="h-2.5 w-2.5 rounded-full border"
              style={{ backgroundColor: on ? colour : "transparent", borderColor: colour }}
              aria-hidden
            />
          )}
          {group.displayName}
          {on && (
            <span className="font-mono text-[10px] font-medium text-slate-300">
              {formatCount(total)}
              {truncated ? "+" : ""}
            </span>
          )}
        </button>
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          aria-expanded={open}
          aria-haspopup="menu"
          aria-label={`${group.displayName} variants`}
          className="flex w-7 items-center justify-center border-l border-border-dark/70 text-slate-400 hover:bg-white/5 hover:text-white"
        >
          <span className="material-symbols-outlined text-[16px]">{open ? "expand_less" : "expand_more"}</span>
        </button>
      </div>

      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full mt-1 w-60 rounded-lg border border-border-dark bg-surface-dark/95 p-1 shadow-xl backdrop-blur-md"
        >
          {ids.map((id) => (
            <VariantRow key={id} id={id} bar={bar} />
          ))}
        </div>
      )}
    </div>
  );
}

function VariantRow({ id, bar }: { id: InstrumentId; bar: InstrumentBarProps }) {
  const inst = INSTRUMENTS[id];
  const count = bar.footprintCounts[id];
  const isLoading = bar.footprintsLoading[id] ?? false;
  const isLoaded = !!count;
  const isVisible = bar.instrumentVisibility[id];

  const onRow = () => {
    if (isLoading) return;
    if (!isLoaded) load(bar, id);
    else bar.onToggleInstrument(id, !isVisible);
  };

  return (
    <div role="menuitem" className="flex items-center gap-1">
      <button
        type="button"
        onClick={onRow}
        disabled={isLoading}
        aria-pressed={isLoaded && isVisible}
        className={`flex min-h-9 flex-1 items-center gap-2 rounded-md px-2 text-left text-[11px] transition-colors ${
          isLoaded && isVisible ? "bg-white/10 text-white" : "text-slate-300 hover:bg-white/5"
        } ${isLoading ? "cursor-wait opacity-60" : ""}`}
      >
        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: inst.color }} aria-hidden />
        <span className="flex-1">{inst.subLabel}</span>
        <span className="font-mono text-[10px] text-slate-400">
          {isLoading
            ? "loading…"
            : isLoaded
              ? `${count.count}${count.truncated ? ` / ${count.total}` : ""}`
              : "Load"}
        </span>
        {isLoaded && !isLoading && (
          <span className="material-symbols-outlined text-[16px] text-slate-400">
            {isVisible ? "visibility" : "visibility_off"}
          </span>
        )}
      </button>
      {isLoaded && !isLoading && (
        <button
          type="button"
          onClick={() => load(bar, id)}
          title="Reload for the current view"
          aria-label={`Reload ${inst.subLabel} for the current view`}
          className="flex h-9 w-9 items-center justify-center rounded-md text-slate-400 hover:bg-white/5 hover:text-white"
        >
          <span className="material-symbols-outlined text-[16px]">refresh</span>
        </button>
      )}
    </div>
  );
}

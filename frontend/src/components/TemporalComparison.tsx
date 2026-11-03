import { useEffect, useState } from "react";
import ViewerShell from "./viewer/ViewerShell";
import QuickviewImage from "./inspector/widgets/QuickviewImage";

/* =========================================================
 * Types
 * =======================================================*/
interface TemporalComparisonProps {
  lat: number;
  lon: number;
  /** Pre-select instrument from the Inspector context */
  initialInstrument?: string;
  onClose: () => void;
}

type TemporalProduct = {
  product_id: string;
  date: string;
  full_date: string;
};

type TemporalPair = {
  product_a: TemporalProduct;
  product_b: TemporalProduct;
  time_gap_info: string;
};

type TemporalResult = {
  pairs: TemporalPair[];
  total_products: number;
  error?: string;
};

/** All instruments supported by the backend temporal endpoint */
const ALL_INSTRUMENTS = [
  "HIRISE",
  "CTX",
  "CRISM",
  "CRISM_TRR3",
  "SHARAD",
  "SHARAD_HIGHRES",
  "HIRISE_DTM",
] as const;

type TemporalInstrument = (typeof ALL_INSTRUMENTS)[number];

/** Short display labels for instrument buttons */
const INSTRUMENT_LABELS: Record<TemporalInstrument, string> = {
  HIRISE: "HiRISE",
  CTX: "CTX",
  CRISM: "CRISM",
  CRISM_TRR3: "CRISM TRR3",
  SHARAD: "SHARAD",
  SHARAD_HIGHRES: "SHARAD HR",
  HIRISE_DTM: "HiRISE DTM",
};

function resolveInitialInstrument(raw?: string): TemporalInstrument {
  if (!raw) return "HIRISE";
  const upper = raw.toUpperCase();
  for (const inst of ALL_INSTRUMENTS) {
    if (inst === upper) return inst;
  }
  return "HIRISE";
}

/* =========================================================
 * TemporalComparison Component
 * =======================================================*/
export default function TemporalComparison({
  lat,
  lon,
  initialInstrument,
  onClose,
}: TemporalComparisonProps) {
  const [result, setResult] = useState<TemporalResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [instrument, setInstrument] = useState<TemporalInstrument>(
    resolveInitialInstrument(initialInstrument),
  );
  const [selectedPair, setSelectedPair] = useState<TemporalPair | null>(null);

  // Fetch temporal pairs
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function fetchPairs() {
      setLoading(true);
      setError(null);
      setResult(null);
      setSelectedPair(null);

      try {
        const res = await fetch("/api/temporal/find_pairs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lat,
            lon,
            radius_km: 50,
            instrument,
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const data: TemporalResult = await res.json();
        if (cancelled) return;
        if (data.error) {
          setError(data.error);
        }
        setResult(data);
      } catch (e: unknown) {
        if (!cancelled && e instanceof Error && e.name !== "AbortError") {
          setError(e.message ?? "Failed to fetch temporal pairs");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchPairs();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [lat, lon, instrument]);

  return (
    <ViewerShell
      variant="modal"
      maxWidthClass="max-w-5xl"
      icon="compare"
      title="Temporal change"
      subtitle={`${lat.toFixed(4)}°, ${lon.toFixed(4)}°E · repeat observations within 50 km`}
      onClose={onClose}
    >
      <div className="flex h-full max-h-[calc(90vh-3.5rem)] flex-col">
        {/* Instrument selector */}
        <div className="px-5 py-3 border-b border-[#232f48] flex items-center gap-3 flex-wrap">
          <span className="text-[10px] text-[#6b7c9c] uppercase font-bold">Instrument</span>
          <div className="flex gap-1.5 flex-wrap">
            {ALL_INSTRUMENTS.map((inst) => (
              <button
                key={inst}
                onClick={() => setInstrument(inst)}
                className={`px-3 py-1.5 rounded text-[10px] font-medium transition-colors ${
                  instrument === inst
                    ? "bg-amber-500/20 border border-amber-500/50 text-amber-400"
                    : "bg-[#1a2333] border border-[#232f48] text-[#92a4c9] hover:border-amber-500/30"
                }`}
              >
                {INSTRUMENT_LABELS[inst]}
              </button>
            ))}
          </div>
          {result && (
            <span className="text-[10px] text-[#6b7c9c] ml-auto font-mono">
              {result.total_products} products found
            </span>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto scrollbar-dark">
          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-20">
              <span className="material-symbols-outlined animate-spin text-3xl text-amber-400 mb-3">
                progress_activity
              </span>
              <p className="text-sm text-slate-400">Finding temporal pairs...</p>
              <p className="text-[10px] text-slate-600 mt-1">
                Querying ODE for {INSTRUMENT_LABELS[instrument]} products near this location
              </p>
            </div>
          )}

          {/* Error */}
          {error && !loading && (
            <div className="p-6">
              <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-center">
                <span className="material-symbols-outlined text-2xl text-red-400 mb-2 block">
                  error
                </span>
                <p className="text-sm text-red-400 mb-1">Query failed</p>
                <p className="text-[11px] text-slate-500">{error}</p>
              </div>
            </div>
          )}

          {/* No pairs found */}
          {result && result.pairs.length === 0 && !loading && !error && (
            <div className="flex flex-col items-center justify-center py-20">
              <span className="material-symbols-outlined text-4xl text-[#3a4a68] mb-3">
                search_off
              </span>
              <p className="text-sm text-slate-400 mb-1">No temporal pairs found</p>
              <p className="text-[11px] text-slate-500 text-center max-w-xs">
                No repeat {INSTRUMENT_LABELS[instrument]} observations were found within 50 km of this location.
                {result.total_products > 0
                  ? ` Found ${result.total_products} product(s) but all on the same date.`
                  : " Try a different instrument or location."}
              </p>
            </div>
          )}

          {/* Pairs list or comparison view */}
          {result && result.pairs.length > 0 && !loading && !selectedPair && (
            <div className="p-5 space-y-3">
              <h4 className="text-[#92a4c9] text-[10px] font-bold uppercase tracking-widest mb-2">
                Temporal Pairs ({result.pairs.length})
              </h4>
              {result.pairs.map((pair, i) => (
                <div
                  key={i}
                  className="rounded-lg border border-[#232f48] bg-[#0a0f18] p-4 hover:border-amber-500/30 transition-colors"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="material-symbols-outlined text-sm text-amber-400">
                        schedule
                      </span>
                      <span className="text-[11px] text-amber-300 font-medium">
                        {pair.time_gap_info}
                      </span>
                    </div>
                    <button
                      onClick={() => setSelectedPair(pair)}
                      className="px-3 py-1.5 rounded text-[10px] font-medium bg-amber-500/20 border border-amber-500/30 text-amber-400 hover:bg-amber-500/30 transition-colors"
                    >
                      Compare
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded bg-[#1a2333] p-2.5 border border-[#232f48]">
                      <p className="text-[9px] text-[#6b7c9c] uppercase font-bold mb-1">
                        Earlier
                      </p>
                      <p className="text-[10px] font-mono text-[#92a4c9] truncate">
                        {pair.product_a.product_id}
                      </p>
                      <p className="text-[9px] text-slate-500 mt-0.5">
                        {pair.product_a.date}
                      </p>
                    </div>
                    <div className="rounded bg-[#1a2333] p-2.5 border border-[#232f48]">
                      <p className="text-[9px] text-[#6b7c9c] uppercase font-bold mb-1">
                        Later
                      </p>
                      <p className="text-[10px] font-mono text-[#92a4c9] truncate">
                        {pair.product_b.product_id}
                      </p>
                      <p className="text-[9px] text-slate-500 mt-0.5">
                        {pair.product_b.date}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Comparison view */}
          {selectedPair && (
            <div className="p-5 space-y-4">
              {/* Back button */}
              <button
                onClick={() => setSelectedPair(null)}
                className="flex items-center gap-1.5 text-[11px] text-[#92a4c9] hover:text-white transition-colors"
              >
                <span className="material-symbols-outlined text-sm">arrow_back</span>
                Back to pairs list
              </button>

              <h4 className="text-white text-sm font-bold flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-400">compare</span>
                Side-by-Side Comparison
              </h4>

              <div className="text-[10px] text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded px-3 py-2">
                {selectedPair.time_gap_info}
              </div>

              <div className="grid grid-cols-2 gap-4">
                {/* Earlier observation */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 font-bold uppercase">
                      Earlier
                    </span>
                    <span className="text-[9px] text-slate-500">
                      {selectedPair.product_a.date}
                    </span>
                  </div>
                  <div className="rounded-lg border border-[#232f48] overflow-hidden bg-black h-80">
                    <QuickviewImage productId={selectedPair.product_a.product_id} instrument={instrument} />
                  </div>
                  <p className="text-[9px] font-mono text-[#6b7c9c] truncate">
                    {selectedPair.product_a.product_id}
                  </p>
                </div>

                {/* Later observation */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] px-2 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30 font-bold uppercase">
                      Later
                    </span>
                    <span className="text-[9px] text-slate-500">
                      {selectedPair.product_b.date}
                    </span>
                  </div>
                  <div className="rounded-lg border border-[#232f48] overflow-hidden bg-black h-80">
                    <QuickviewImage productId={selectedPair.product_b.product_id} instrument={instrument} />
                  </div>
                  <p className="text-[9px] font-mono text-[#6b7c9c] truncate">
                    {selectedPair.product_b.product_id}
                  </p>
                </div>
              </div>

              {/* Note about future enhancement */}
              <div className="rounded-lg border border-[#232f48] bg-[#0a0f18] p-3 mt-4">
                <div className="flex items-start gap-2">
                  <span className="material-symbols-outlined text-sm text-[#6b7c9c] mt-0.5">
                    info
                  </span>
                  <div>
                    <p className="text-[10px] text-[#92a4c9] font-medium mb-1">
                      Preview Mode
                    </p>
                    <p className="text-[9px] text-[#6b7c9c] leading-relaxed">
                      This is a discovery tool showing quickview thumbnails. Pixel-level
                      differencing and automated change detection will be available in a
                      future release once both products are downloaded at full resolution.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

      </div>
    </ViewerShell>
  );
}

import { useEffect, useState } from "react";
import toast from "react-hot-toast";

type MineralStats = {
  classified_pixels?: number;
  confidence_threshold?: number;
  mean_confidence?: number;
};

type LegendItem = {
  mineral_id: string;
  name: string;
  color_hex: string;
  pixel_count: number;
  avg_confidence?: number;
};

type Status = "checking" | "not_downloaded" | "idle" | "loading" | "done" | "error";

export default function TRR3MineralSection({
  obsId,
  onOpenMineralSequence,
}: {
  obsId: string;
  onOpenMineralSequence?: (obsId: string) => void;
}) {
  const [status, setStatus] = useState<Status>("checking");
  const [stats, setStats] = useState<MineralStats | null>(null);
  const [legend, setLegend] = useState<LegendItem[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [progressMsg, setProgressMsg] = useState("");
  const [progressPct, setProgressPct] = useState<number | null>(null);

  // Check data availability on mount
  useEffect(() => {
    let cancelled = false;

    fetch(`/api/mineral-cnn/result/${obsId}/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return;
        if (data) {
          setStats(data);
          setStatus("done");
          fetch(`/api/mineral-cnn/result/${obsId}/legend`)
            .then((r) => r.json())
            .then((d) => !cancelled && setLegend(d.legend || []))
            .catch(() => {});
          return;
        }
        return fetch(`/api/mineral-cnn/acquire/${obsId}/status`);
      })
      .then((res) => {
        if (!res || cancelled) return;
        return res.json();
      })
      .then((data) => {
        if (!data || cancelled) return;
        if (data.has_results) {
          setStatus("done");
        } else if (data.has_trr3_data) {
          setStatus("idle");
        } else {
          setStatus("not_downloaded");
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setStatus("not_downloaded");
          toast.error(`CNN status check failed: ${e.message}`);
        }
      });

    return () => { cancelled = true; };
  }, [obsId]);

  // SSE stream reader
  const streamSSE = async (url: string) => {
    setStatus("loading");
    setErrorMsg("");
    setProgressMsg("Starting...");
    setProgressPct(null);
    try {
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";
      let gotError = false;
      let errDetail = "Pipeline failed";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.replace(/^data:\s*/, "").trim();
          if (!trimmed.startsWith("{")) continue;
          try {
            const evt = JSON.parse(trimmed);
            if (evt.event === "error") {
              gotError = true;
              errDetail = evt.data?.error || "Pipeline failed";
              break;
            }
            if (evt.event === "status" && evt.data?.message) {
              setProgressMsg(evt.data.message);
              if (evt.data.step !== "jcat") setProgressPct(null);
            }
            if (evt.event === "progress" && evt.data?.percent != null) {
              setProgressPct(evt.data.percent);
              setProgressMsg(`JCAT atmospheric correction: ${evt.data.percent}%`);
            }
            if (evt.event === "download_progress" && evt.data?.percent != null) {
              setProgressPct(evt.data.percent);
              setProgressMsg(`Downloading ${evt.data.file}: ${evt.data.percent}%`);
            }
            if (evt.event === "discovery" && evt.data) {
              setProgressMsg(`Found ${evt.data.files} files (${evt.data.total_size_mb} MB)`);
            }
          } catch { /* skip malformed JSON */ }
        }
        if (gotError) break;
      }

      if (gotError) {
        setStatus("error");
        setErrorMsg(errDetail);
        return;
      }

      // Fetch results
      const statsRes = await fetch(`/api/mineral-cnn/result/${obsId}/stats`);
      if (statsRes.ok) {
        setStats(await statsRes.json());
        setStatus("done");
        const legendRes = await fetch(`/api/mineral-cnn/result/${obsId}/legend`);
        if (legendRes.ok) {
          const ld = await legendRes.json();
          setLegend(ld.legend || []);
        }
      } else {
        await new Promise((r) => setTimeout(r, 500));
        const retry = await fetch(`/api/mineral-cnn/result/${obsId}/stats`);
        if (retry.ok) {
          setStats(await retry.json());
          setStatus("done");
        } else {
          setStatus("error");
          setErrorMsg("Pipeline completed but results not available");
        }
      }
    } catch (e: unknown) {
      setStatus("error");
      const msg = e instanceof Error ? e.message : "Pipeline failed";
      setErrorMsg(msg);
    }
  };

  const runAcquire = () => streamSSE(`/api/mineral-cnn/acquire/${obsId}`);
  const runClassification = () => streamSSE(`/api/mineral-cnn/classify/${obsId}`);

  return (
    <div className="space-y-3">
      <h4 className="text-xs font-bold uppercase tracking-wider text-teal-400 flex items-center gap-1.5">
        <span className="material-symbols-outlined text-sm">science</span>
        CNN Mineral Classification
      </h4>

      {status === "checking" && (
        <div className="flex items-center gap-2 py-2 text-slate-400 text-[11px]">
          <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
          Checking data availability...
        </div>
      )}

      {status === "not_downloaded" && (
        <div className="space-y-2">
          <p className="text-[9px] text-slate-400/80 leading-relaxed">
            TRR3 data not downloaded yet. This will download L-sensor TRR3 + DDR from PDS, then run JCAT atmospheric correction and CNN classification.
          </p>
          <button
            onClick={runAcquire}
            className="w-full px-3 py-2 rounded text-[11px] font-medium bg-teal-500/20 border border-teal-500/30 text-teal-400 hover:bg-teal-500/30 transition-colors flex items-center justify-center gap-2"
          >
            <span className="material-symbols-outlined text-sm">download</span>
            Download & Classify
          </button>
        </div>
      )}

      {status === "idle" && (
        <div className="space-y-2">
          <p className="text-[9px] text-amber-400/80 flex items-center gap-1 leading-relaxed">
            <span className="material-symbols-outlined text-xs">info</span>
            TRR3 data available locally. Ready to classify.
          </p>
          <button
            onClick={runClassification}
            className="w-full px-3 py-2 rounded text-[11px] font-medium bg-teal-500/20 border border-teal-500/30 text-teal-400 hover:bg-teal-500/30 transition-colors flex items-center justify-center gap-2"
          >
            <span className="material-symbols-outlined text-sm">play_arrow</span>
            Run Classification
          </button>
        </div>
      )}

      {status === "loading" && (
        <div className="space-y-2 py-3">
          <div className="flex items-center gap-2 text-teal-400 text-[11px]">
            <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
            <span className="truncate">{progressMsg || "Running CNN inference..."}</span>
          </div>
          {progressPct != null && (
            <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-500 rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          )}
        </div>
      )}

      {status === "error" && (
        <div className="space-y-2">
          <p className="text-[11px] text-red-400 flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">error</span>
            {errorMsg}
          </p>
          <button
            onClick={runAcquire}
            className="px-3 py-1.5 rounded text-[10px] font-medium bg-teal-500/20 border border-teal-500/30 text-teal-400 hover:bg-teal-500/30 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {status === "done" && stats && (
        <div className="space-y-3">
          {/* Mineral map */}
          <div className="overflow-hidden rounded-lg border border-border-dark">
            <img
              src={`/api/mineral-cnn/result/${obsId}/mineral-map.png`}
              alt="Mineral Map"
              className="w-full bg-black"
              loading="lazy"
            />
          </div>

          {/* Stats summary */}
          <div className="grid grid-cols-3 gap-2 text-[10px]">
            <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
              <div className="text-[9px] uppercase text-slate-500">Classified</div>
              <div className="font-mono text-white">{stats.classified_pixels?.toLocaleString()}</div>
            </div>
            <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
              <div className="text-[9px] uppercase text-slate-500">Threshold</div>
              <div className="font-mono text-white">
                ≥{((stats.confidence_threshold ?? 0.95) * 100).toFixed(0)}%
              </div>
            </div>
            <div className="rounded border border-border-dark/50 bg-bg-dark/40 p-2">
              <div className="text-[9px] uppercase text-slate-500">Mean Conf.</div>
              <div
                className={`font-mono ${
                  (stats.mean_confidence ?? 0) >= 0.95
                    ? "text-emerald-400"
                    : (stats.mean_confidence ?? 0) >= 0.8
                      ? "text-amber-400"
                      : "text-red-400"
                }`}
              >
                {((stats.mean_confidence ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Legend */}
          {legend.length > 0 && (
            <div className="space-y-1">
              <h5 className="text-[9px] uppercase text-slate-500 font-bold">Minerals Detected</h5>
              <div className="max-h-40 overflow-y-auto scrollbar-dark space-y-0.5">
                {legend.map((item) => {
                  const conf = item.avg_confidence;
                  const confPct = (conf ?? 0) * 100;
                  const confColor =
                    confPct >= 95 ? "text-emerald-400" : confPct >= 80 ? "text-amber-400" : "text-red-400";
                  return (
                    <div key={item.mineral_id} className="flex items-center gap-2 py-0.5">
                      <span
                        className="w-3 h-3 rounded-sm flex-shrink-0"
                        style={{ backgroundColor: item.color_hex }}
                      />
                      <span className="text-[10px] text-white flex-1 truncate">{item.name}</span>
                      {conf != null && conf > 0 && (
                        <span
                          className={`text-[8px] font-mono px-1 py-0.5 rounded ${confColor} bg-surface-dark/60`}
                        >
                          {confPct.toFixed(0)}%
                        </span>
                      )}
                      <span className="text-[9px] text-slate-500 font-mono">{item.pixel_count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Mineral Sequence button */}
          {onOpenMineralSequence && (
            <button
              onClick={() => onOpenMineralSequence(obsId)}
              className="w-full px-3 py-2 rounded text-[11px] font-medium bg-amber-500/20 border border-amber-500/30 text-amber-400 hover:bg-amber-500/30 transition-colors flex items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined text-sm">science</span>
              Mineral Sequence Analysis
            </button>
          )}
        </div>
      )}
    </div>
  );
}

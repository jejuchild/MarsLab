import { useEffect, useState } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

/* =========================================================
 * Types
 * =======================================================*/
export type ProfilePoint = { lat: number; lon: number };

type TransectSample = {
  distance_km: number;
  elevation_m: number;
  slope_deg: number;
  lat: number;
  lon: number;
};

type TransectResponse = {
  profile: TransectSample[];
  total_distance_km: number;
  num_points: number;
  start: { lat: number; lon: number };
  end: { lat: number; lon: number };
};

/* =========================================================
 * Helpers
 * =======================================================*/
function fmtCoord(lat: number, lon: number): string {
  const latStr = `${Math.abs(lat).toFixed(4)}\u00b0${lat >= 0 ? "N" : "S"}`;
  const lonStr = `${Math.abs(lon).toFixed(4)}\u00b0${lon >= 0 ? "E" : "W"}`;
  return `${latStr}, ${lonStr}`;
}

/** Map slope value to a color class for the slope legend */
function slopeColor(deg: number): string {
  if (deg < 3) return "#22c55e";   // green
  if (deg < 5) return "#eab308";   // yellow
  return "#ef4444";                 // red
}

/* Custom tooltip for dual-axis chart */
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || payload.length === 0) return null;

  const elevation = payload.find((p: any) => p.dataKey === "elevation_m");
  const slope = payload.find((p: any) => p.dataKey === "slope_deg");
  const sample = payload[0]?.payload as TransectSample | undefined;
  const slopeDeg = slope?.value ?? 0;

  return (
    <div className="bg-[#0a0f18] border border-[#232f48] rounded-md px-3 py-2 text-[11px] shadow-xl">
      <p className="text-slate-500 mb-1">{(label as number).toFixed(1)} km</p>
      {elevation && (
        <p className="text-blue-400">
          Elevation: <span className="font-mono">{(elevation.value as number).toFixed(0)} m</span>
        </p>
      )}
      {slope && (
        <p style={{ color: slopeColor(slopeDeg) }}>
          Slope: <span className="font-mono">{slopeDeg.toFixed(1)}&deg;</span>
        </p>
      )}
      {sample && (
        <p className="text-slate-600 mt-1 text-[9px]">
          {fmtCoord(sample.lat, sample.lon)}
        </p>
      )}
    </div>
  );
}

/* =========================================================
 * LineProfile Popup Component
 * =======================================================*/
export default function LineProfile({
  startPoint,
  endPoint,
  onClose,
}: {
  startPoint: ProfilePoint;
  endPoint: ProfilePoint;
  onClose: () => void;
}) {
  const [data, setData] = useState<TransectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function fetchProfile() {
      setLoading(true);
      setError(null);
      setData(null);

      try {
        const params = new URLSearchParams({
          start_lat: startPoint.lat.toString(),
          start_lon: startPoint.lon.toString(),
          end_lat: endPoint.lat.toString(),
          end_lon: endPoint.lon.toString(),
          num_points: "300",
        });

        // Try the enhanced transect endpoint first
        let res = await fetch(`/api/terrain/transect_profile?${params}`, {
          signal: controller.signal,
        });

        if (!res.ok) {
          // Fallback to original line_profile endpoint
          const fallbackParams = new URLSearchParams({
            start_lat: startPoint.lat.toString(),
            start_lon: startPoint.lon.toString(),
            end_lat: endPoint.lat.toString(),
            end_lon: endPoint.lon.toString(),
            num_samples: "300",
          });
          res = await fetch(`/api/terrain/line_profile?${fallbackParams}`, {
            signal: controller.signal,
          });
        }

        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.error || `HTTP ${res.status}`);
        }

        const json = await res.json();

        // Normalize response from either endpoint
        const profile: TransectSample[] = json.profile.map((p: any) => ({
          distance_km: p.distance_km,
          elevation_m: p.elevation_m ?? 0,
          slope_deg: p.slope_deg ?? 0,
          lat: p.lat,
          lon: p.lon,
        }));

        if (cancelled) return;
        setData({
          profile,
          total_distance_km: json.total_distance_km,
          num_points: profile.length,
          start: json.start,
          end: json.end,
        });
      } catch (e: unknown) {
        if (!cancelled && e instanceof Error && e.name !== "AbortError") {
          setError(e.message ?? "Unknown error");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchProfile();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [startPoint.lat, startPoint.lon, endPoint.lat, endPoint.lon]);

  // Compute elevation range for chart domain
  const elevations = data?.profile.map((p) => p.elevation_m) ?? [];
  const minElev = elevations.length > 0 ? Math.floor(Math.min(...elevations) / 100) * 100 : 0;
  const maxElev = elevations.length > 0 ? Math.ceil(Math.max(...elevations) / 100) * 100 : 0;

  // Compute slope range
  const slopes = data?.profile.map((p) => p.slope_deg) ?? [];
  const maxSlope = slopes.length > 0 ? Math.ceil(Math.max(...slopes)) : 10;
  const slopeDomain = [0, Math.max(maxSlope, 5)];

  // Stats summary
  const meanSlope = slopes.length > 0 ? slopes.reduce((a, b) => a + b, 0) / slopes.length : 0;
  const elevRange = elevations.length > 0 ? Math.max(...elevations) - Math.min(...elevations) : 0;

  return (
    <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-40 w-[780px] max-w-[92vw]">
      <div className="bg-[#101622] border border-[#232f48] rounded-xl shadow-2xl shadow-black/50 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#232f48] bg-[#0a0f18]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-base">show_chart</span>
            <h3 className="text-white text-xs font-bold uppercase tracking-wider">
              Elevation &amp; Slope Profile
            </h3>
            {data && (
              <span className="text-[10px] text-slate-500 font-mono ml-2">
                {data.total_distance_km.toFixed(1)} km
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-500 hover:text-red-400 transition-colors"
            title="Close profile"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        {/* Coordinates + Stats */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-[#232f48]/50 text-[10px]">
          <div className="flex gap-4">
            <div>
              <span className="text-slate-500 uppercase font-bold mr-1.5">Start:</span>
              <span className="text-primary font-mono">{fmtCoord(startPoint.lat, startPoint.lon)}</span>
            </div>
            <div>
              <span className="text-slate-500 uppercase font-bold mr-1.5">End:</span>
              <span className="text-primary font-mono">{fmtCoord(endPoint.lat, endPoint.lon)}</span>
            </div>
          </div>
          {data && (
            <div className="flex gap-3">
              <span className="text-slate-500">
                Elev range: <span className="text-white font-mono">{elevRange.toFixed(0)} m</span>
              </span>
              <span className="text-slate-500">
                Mean slope: <span className="text-white font-mono">{meanSlope.toFixed(1)}&deg;</span>
              </span>
            </div>
          )}
        </div>

        {/* Legend */}
        {data && !loading && (
          <div className="flex items-center gap-4 px-4 py-1.5 border-b border-[#232f48]/30 text-[9px]">
            <div className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-blue-500 rounded" />
              <span className="text-slate-500">Elevation</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="inline-block w-3 h-2 bg-orange-500/30 rounded-sm" />
              <span className="text-slate-500">Slope</span>
            </div>
            <div className="flex items-center gap-1 ml-auto">
              <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
              <span className="text-slate-500">&lt;3&deg;</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="inline-block w-2 h-2 rounded-full bg-yellow-500" />
              <span className="text-slate-500">3-5&deg;</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
              <span className="text-slate-500">&ge;5&deg;</span>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="px-4 py-3" style={{ height: 260 }}>
          {loading && (
            <div className="flex flex-col items-center justify-center h-full">
              <span className="material-symbols-outlined animate-spin text-2xl text-primary mb-2">
                progress_activity
              </span>
              <p className="text-xs text-slate-400">Computing transect profile...</p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center h-full">
              <span className="material-symbols-outlined text-2xl text-red-400 mb-2">error</span>
              <p className="text-xs text-red-400">{error}</p>
            </div>
          )}

          {data && !loading && (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={data.profile}
                margin={{ top: 5, right: 50, left: 10, bottom: 5 }}
              >
                <defs>
                  <linearGradient id="slopeGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#f97316" stopOpacity={0.05} />
                  </linearGradient>
                </defs>

                <CartesianGrid strokeDasharray="3 3" stroke="#232f48" />

                <XAxis
                  dataKey="distance_km"
                  type="number"
                  domain={[0, "dataMax"]}
                  tick={{ fontSize: 10, fill: "#6b7c9c" }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}`}
                  label={{
                    value: "Distance (km)",
                    position: "insideBottom",
                    offset: -2,
                    style: { fontSize: 10, fill: "#6b7c9c" },
                  }}
                />

                {/* Left Y-axis: Elevation */}
                <YAxis
                  yAxisId="elev"
                  dataKey="elevation_m"
                  domain={[minElev, maxElev]}
                  tick={{ fontSize: 10, fill: "#6b7c9c" }}
                  tickFormatter={(v: number) => `${(v / 1000).toFixed(1)}`}
                  label={{
                    value: "Elev (km)",
                    angle: -90,
                    position: "insideLeft",
                    offset: 5,
                    style: { fontSize: 10, fill: "#6b7c9c" },
                  }}
                />

                {/* Right Y-axis: Slope */}
                <YAxis
                  yAxisId="slope"
                  orientation="right"
                  domain={slopeDomain}
                  tick={{ fontSize: 10, fill: "#f9731680" }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}\u00b0`}
                  label={{
                    value: "Slope (\u00b0)",
                    angle: 90,
                    position: "insideRight",
                    offset: -5,
                    style: { fontSize: 10, fill: "#f9731680" },
                  }}
                />

                {/* Slope threshold reference lines */}
                <ReferenceLine yAxisId="slope" y={3} stroke="#22c55e" strokeDasharray="4 4" strokeOpacity={0.3} />
                <ReferenceLine yAxisId="slope" y={5} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.3} />

                <Tooltip content={<CustomTooltip />} />

                {/* Slope filled area */}
                <Area
                  yAxisId="slope"
                  type="monotone"
                  dataKey="slope_deg"
                  fill="url(#slopeGradient)"
                  stroke="#f97316"
                  strokeWidth={0.5}
                  strokeOpacity={0.5}
                  dot={false}
                  activeDot={false}
                  isAnimationActive={false}
                />

                {/* Elevation line */}
                <Line
                  yAxisId="elev"
                  type="monotone"
                  dataKey="elevation_m"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 3, fill: "#3b82f6" }}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

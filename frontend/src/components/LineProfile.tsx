import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/* =========================================================
 * Types
 * =======================================================*/
export type ProfilePoint = { lat: number; lon: number };

type ProfileSample = {
  distance_km: number;
  elevation_m: number | null;
  lat: number;
  lon: number;
};

type ProfileResponse = {
  profile: ProfileSample[];
  total_distance_km: number;
  num_samples: number;
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
  const [data, setData] = useState<ProfileResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

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
          num_samples: "300",
        });
        const res = await fetch(`/terrain/line_profile?${params}`, {
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          throw new Error(body?.error || `HTTP ${res.status}`);
        }
        const json: ProfileResponse = await res.json();
        setData(json);
      } catch (e: any) {
        if (e.name !== "AbortError") {
          setError(e.message ?? "Unknown error");
        }
      } finally {
        setLoading(false);
      }
    }

    fetchProfile();
    return () => controller.abort();
  }, [startPoint.lat, startPoint.lon, endPoint.lat, endPoint.lon]);

  // Compute elevation range for chart domain
  const elevations = data?.profile
    .map((p) => p.elevation_m)
    .filter((v): v is number => v !== null) ?? [];
  const minElev = elevations.length > 0 ? Math.floor(Math.min(...elevations) / 100) * 100 : 0;
  const maxElev = elevations.length > 0 ? Math.ceil(Math.max(...elevations) / 100) * 100 : 0;

  return (
    <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-40 w-[720px] max-w-[90vw]">
      <div className="bg-[#101622] border border-[#232f48] rounded-xl shadow-2xl shadow-black/50 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#232f48] bg-[#0a0f18]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-base">show_chart</span>
            <h3 className="text-white text-xs font-bold uppercase tracking-wider">
              Elevation Profile
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

        {/* Coordinates */}
        <div className="flex gap-4 px-4 py-2 border-b border-[#232f48]/50 text-[10px]">
          <div>
            <span className="text-slate-500 uppercase font-bold mr-1.5">Start:</span>
            <span className="text-primary font-mono">{fmtCoord(startPoint.lat, startPoint.lon)}</span>
          </div>
          <div>
            <span className="text-slate-500 uppercase font-bold mr-1.5">End:</span>
            <span className="text-primary font-mono">{fmtCoord(endPoint.lat, endPoint.lon)}</span>
          </div>
        </div>

        {/* Content */}
        <div className="px-4 py-3" style={{ height: 240 }}>
          {loading && (
            <div className="flex flex-col items-center justify-center h-full">
              <span className="material-symbols-outlined animate-spin text-2xl text-primary mb-2">
                progress_activity
              </span>
              <p className="text-xs text-slate-400">Computing elevation profile...</p>
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
              <LineChart
                data={data.profile.filter((p) => p.elevation_m !== null)}
                margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
              >
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
                <YAxis
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
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0a0f18",
                    border: "1px solid #232f48",
                    borderRadius: 6,
                    fontSize: 11,
                  }}
                  labelStyle={{ color: "#6b7c9c" }}
                  formatter={(value?: number) => [`${(value ?? 0).toFixed(0)} m`, "Elevation"]}
                  labelFormatter={(label: number) => `${label.toFixed(1)} km`}
                />
                <Line
                  type="monotone"
                  dataKey="elevation_m"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 3, fill: "#3b82f6" }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

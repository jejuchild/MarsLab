// src/components/detail-inspector/panels/crism/CRISMSpectrumPanel.tsx
import { useEffect, useState } from "react";

type Props = {
  productId: string;
  lat: number;
  lon: number;
};

type SpectrumResponse = {
  wavelength: number[]; // microns
  spectrum: number[];   // I/F
};

export default function CRISMSpectrumPanel({
  productId,
  lat,
  lon,
}: Props) {
  const [data, setData] = useState<SpectrumResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!productId) return;

    const controller = new AbortController();

    async function fetchSpectrum() {
      setLoading(true);
      setError(null);
      setData(null);

      try {
        const url =
          `/crism/${productId}/spectrum` +
          `?lat=${lat}&lon=${lon}`;

        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const json = (await res.json()) as SpectrumResponse;
        setData(json);
      } catch (e: any) {
        if (e.name !== "AbortError") {
          console.error("[CRISMSpectrumPanel]", e);
          setError("Failed to load spectrum");
        }
      } finally {
        setLoading(false);
      }
    }

    fetchSpectrum();
    return () => controller.abort();
  }, [productId, lat, lon]);

  // -------------------------
  // Render
  // -------------------------
  if (loading) {
    return <div style={styles.placeholder}>Loading spectrum…</div>;
  }

  if (error) {
    return (
      <div style={{ ...styles.placeholder, color: "#ff6b6b" }}>
        {error}
      </div>
    );
  }

  if (!data) {
    return <div style={styles.placeholder}>No spectrum</div>;
  }

  return (
    <div style={styles.container}>
      <SpectrumSVG
        wavelength={data.wavelength}
        spectrum={data.spectrum}
      />
    </div>
  );
}

// ==================================================
// SVG Spectrum Renderer (가볍고 빠름)
// ==================================================
function SpectrumSVG({
  wavelength,
  spectrum,
}: {
  wavelength: number[];
  spectrum: number[];
}) {
  const W = 600;
  const H = 220;
  const PAD = 30;

  const xMin = Math.min(...wavelength);
  const xMax = Math.max(...wavelength);
  const yMin = Math.min(...spectrum);
  const yMax = Math.max(...spectrum);

  const xScale = (x: number) =>
    PAD + ((x - xMin) / (xMax - xMin)) * (W - 2 * PAD);

  const yScale = (y: number) =>
    H - PAD - ((y - yMin) / (yMax - yMin)) * (H - 2 * PAD);

  const path = wavelength
    .map((w, i) => {
      const x = xScale(w);
      const y = yScale(spectrum[i]);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${W} ${H}`}
      style={{ background: "#000" }}
    >
      {/* axes */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#666" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#666" />

      {/* spectrum */}
      <path d={path} fill="none" stroke="#4fc3f7" strokeWidth={1.5} />

      {/* labels */}
      <text x={W / 2} y={H - 5} fill="#aaa" fontSize={10} textAnchor="middle">
        Wavelength (µm)
      </text>
      <text
        x={5}
        y={H / 2}
        fill="#aaa"
        fontSize={10}
        textAnchor="middle"
        transform={`rotate(-90 5 ${H / 2})`}
      >
        I/F
      </text>
    </svg>
  );
}

// -------------------------
// Styles
// -------------------------
const styles: Record<string, React.CSSProperties> = {
  container: {
    width: "100%",
    height: "100%",
    background: "#000",
    padding: 4,
  },
  placeholder: {
    width: "100%",
    height: "100%",
    background: "#111",
    color: "#aaa",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 13,
  },
};

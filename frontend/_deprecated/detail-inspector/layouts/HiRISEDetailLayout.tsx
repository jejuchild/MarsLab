import type { HiRISEWindowResponse } from "../../../types/hirise";

import QuickviewPanel from "../panels/QuickviewPanel";
import HiRISEImagePanel from "../panels/hirise/HiRISEImagePanel";
import HiRISEHistogramPanel from "../panels/hirise/HiRISEHistogramPanel";
import HiRISEStatsPanel from "../panels/hirise/HiRISEStatsPanel";

/* =========================================================
 * Props
 * =======================================================*/
type Props = {
  productId: string;

  pin: { x: number; y: number } | null;
  setPin: React.Dispatch<
    React.SetStateAction<{ x: number; y: number } | null>
  >;

  analysis: {
    running: boolean;
    error: string | null;
    hiriseData: HiRISEWindowResponse | null;
    runHiRISEAnalysis: (
      productId: string,
      pin: { x: number; y: number } | null
    ) => void;
  };
};

/* =========================================================
 * HiRISE Detail Layout
 * =======================================================*/
export default function HiRISEDetailLayout({
  productId,
  pin,
  setPin,
  analysis,
}: Props) {
  const { running, error, hiriseData, runHiRISEAnalysis } = analysis;

  return (
    <>
      {/* ================= LEFT: QUICKVIEW ================= */}
      <QuickviewPanel productId={productId} />

      {/* ================= RIGHT ================= */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* ===== IMAGE ===== */}
        <HiRISEImagePanel
          productId={productId}
          pin={pin}
          onDoubleClick={(xy) => {
            setPin((prev) => (prev ? null : xy));
          }}
        />

        {/* ===== ANALYSIS ===== */}
        <div
          style={{
            flex: "0 0 30%",
            minHeight: 0,
            borderTop: "1px solid #333",
            padding: 10,
            background: "#0d0d0d",
          }}
        >
          <button
            onClick={() => runHiRISEAnalysis(productId, pin)}
            disabled={running}
            style={{
              background: "#1e1e1e",
              border: "1px solid #333",
              color: "#fff",
              padding: "4px 8px",
              fontSize: 11,
              marginBottom: 8,
            }}
          >
            {running ? "Running analysis…" : "Run HiRISE Analysis"}
          </button>

          {error && <div style={{ color: "#f66" }}>{error}</div>}

          {hiriseData && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "2fr 1fr",
                gap: 12,
                height: "100%",
              }}
            >
              <HiRISEHistogramPanel
                counts={hiriseData.histogram.counts}
                centerBin={hiriseData.histogram.center_bin}
              />

              <HiRISEStatsPanel
                stats={hiriseData.stats}
                center={hiriseData.center}
              />
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// detail-inspector/layouts/CRISMDetailLayout.tsx
import { useState } from "react";

import CRISMQuickviewPanel from "../panels/crism/CRISMQuickviewPanel";
import CRISMImagePanel from "../panels/crism/CRISMImagePanel";
import CRISMSpectrumPanel from "../panels/crism/CRISMSpectrumPanel";

type Props = {
  productId: string;
};

export default function CRISMDetailLayout({ productId }: Props) {
  const [clicked, setClicked] = useState<{
    lat: number;
    lon: number;
  } | null>(null);

  return (
    <>
      {/* ================= LEFT: QUICKVIEW (SMALL) ================= */}
      <div
        style={{
          width: 220,
          borderRight: "1px solid #333",
          padding: 8,
          background: "#111",
        }}
      >
        <CRISMQuickviewPanel productId={productId} />
      </div>

      {/* ================= RIGHT ================= */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* ===== IMAGE + CONTROLS ===== */}
        <div
          style={{
            flex: 1,
            display: "flex",
            borderBottom: "1px solid #333",
          }}
        >
          {/* IMAGE */}
          <div style={{ flex: 1 }}>
            <CRISMImagePanel
              productId={productId}
              onClick={(lat, lon) => {
                setClicked({ lat, lon });
              }}
            />
          </div>

          {/* CONTROLS (placeholder) */}
          <div
            style={{
              width: 220,
              borderLeft: "1px solid #333",
              padding: 8,
              fontSize: 12,
              opacity: 0.7,
            }}
          >
            <div>Browse type</div>
            <ul>
              <li>VNIR Albedo</li>
              <li>Hydrated minerals</li>
              <li>Ice / IC2</li>
            </ul>
          </div>
        </div>

        {/* ===== SPECTRUM ===== */}
        <div
          style={{
            height: "30%",
            minHeight: 180,
            padding: 8,
            background: "#0d0d0d",
          }}
        >
          {clicked ? (
            <CRISMSpectrumPanel
              productId={productId}
              lat={clicked.lat}
              lon={clicked.lon}
            />
          ) : (
            <div style={{ opacity: 0.6, fontSize: 12 }}>
              Click on image to view spectrum
            </div>
          )}
        </div>
      </div>
    </>
  );
}

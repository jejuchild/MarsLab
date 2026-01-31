// detail-inspector/panels/crism/CRISMQuickviewPanel.tsx
import React from "react";

/* =========================================================
 * Props
 * =======================================================*/
type Props = {
  productId: string;
  obsId?: string;
};

/* =========================================================
 * CRISM Quickview Panel
 * - uses browse quickview (brvnaj)
 * =======================================================*/
export default function CRISMQuickviewPanel({ productId }: Props) {
  // ✅ 안전 가드 (렌더 타이밍 문제 방지)
  if (!productId) {
    return (
      <div
        style={{
          width: 220,
          borderRight: "1px solid #333",
          padding: 8,
          background: "#111",
          color: "#666",
          fontSize: 12,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        No CRISM product
      </div>
    );
  }

  // CRISM productId → browse quickview filename conversion
  // Standard: frt00007d87_07_if164j_mtr3 -> frt00007d87_07_brvnaj_mtr3.png
  // Arcadia:  frt00003156_07_brcarj_mtr3 -> frt00003156_VNIR.png
  let src: string;
  if (productId.includes("_brcarj_")) {
    const baseObsId = productId.split("_")[0];
    src = `/crism/quickview/${baseObsId}_VNIR.png`;
  } else {
    const quickviewName = productId.replace(/_if[0-9a-z]+_mtr3$/i, "_brvnaj_mtr3.png");
    src = `/crism/quickview/${quickviewName}`;
  }

  return (
    <div
      style={{
        width: 220,
        borderRight: "1px solid #333",
        padding: 8,
        display: "flex",
        flexDirection: "column",
        background: "#111",
      }}
    >
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 6 }}>
        CRISM Quickview
      </div>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#000",
          border: "1px solid #222",
        }}
      >
        <img
          src={src}
          style={{
            maxWidth: "100%",
            maxHeight: "100%",
            objectFit: "contain",
            display: "block",
          }}
          onError={(e) => {
            console.warn("[CRISM quickview missing]", src);
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    </div>
  );
}

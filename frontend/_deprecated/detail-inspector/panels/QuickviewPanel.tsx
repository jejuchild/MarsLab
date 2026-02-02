// src/components/detail-inspector/QuickviewPanel.tsx
import React from "react";

export default function QuickviewPanel({ productId }: { productId: string }) {
  return (
    <div
      style={{
        width: 260,
        borderRight: "1px solid #333",
        padding: 8,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 4 }}>
        HiRISE Quickview
      </div>

      <div
        style={{
          position: "relative",
          flex: 1,
          minHeight: 0,
        }}
      >
        <img
          src={`/hirise/quickview/${productId}.jpg`}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: "block",
            background: "#000",
          }}
        />
      </div>
    </div>
  );
}

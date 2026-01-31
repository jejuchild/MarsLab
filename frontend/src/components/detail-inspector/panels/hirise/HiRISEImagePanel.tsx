// src/components/detail-inspector//panels/hirise/HiRISEImagePanel.tsx
import React from "react";
import WorldTileViewer from "../../../WorldTileViewer";

export default function HiRISEImagePanel({
  productId,
  pin,
  onDoubleClick,
}: {
  productId: string;
  pin: { x: number; y: number } | null;
  onDoubleClick: (xy: { x: number; y: number }) => void;
}) {
  return (
    <div style={{ flex: 1 }}>
      <WorldTileViewer
        productId={productId}
        pin={pin}
        onDoubleClick={onDoubleClick}
      />
    </div>
  );
}

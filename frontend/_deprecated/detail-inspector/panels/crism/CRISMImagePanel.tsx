// src/components/detail-inspector/panels/crism/CRISMImagePanel.tsx
import { useEffect, useState } from "react";

type Props = {
  productId?: string;
};

export default function CRISMImagePanel({ productId }: Props) {
      console.log("[CRISMImagePanel] render", productId);

      if (!productId) {
    return (
      <div style={styles.placeholder}>
        No CRISM product selected
      </div>
    );
  }

  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {

    if (!productId) return;

    const controller = new AbortController();

    async function fetchRGB() {
      setLoading(true);
      setError(null);
      setImgUrl(null);

      try {
        const res = await fetch(
          `/crism/${productId}/rgb`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              r_um: 2.53,
              g_um: 1.50,
              b_um: 1.08,
              vmin: 0.02,
              vmax: 0.25,
            }),
            signal: controller.signal,
          }
        );

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }

        const blob = await res.blob();
        console.log("[CRISMImagePanel] blob", blob, blob.size);
        const url = URL.createObjectURL(blob);
        setImgUrl(url);
      } catch (e: any) {
        if (e.name !== "AbortError") {
          console.error("[CRISMImagePanel]", e);
          setError("Failed to load CRISM RGB image");
        }
      } finally {
        setLoading(false);
      }
    }

    fetchRGB();

    return () => {
      controller.abort();
      if (imgUrl) URL.revokeObjectURL(imgUrl);
    };
  }, [productId]);

  // -------------------------
  // Render
  // -------------------------
  if (loading) {
    return <div style={styles.placeholder}>Loading CRISM RGB…</div>;
  }

  if (error) {
    return (
      <div style={{ ...styles.placeholder, color: "#ff6b6b" }}>
        {error}
      </div>
    );
  }

  if (!imgUrl) {
    return <div style={styles.placeholder}>No image</div>;
  }

  return (
    <div style={styles.container}>
      <img
        src={imgUrl}
        alt={`CRISM RGB ${productId}`}
        style={styles.image}
      />
    </div>
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
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  image: {
    maxWidth: "100%",
    maxHeight: "100%",
    imageRendering: "pixelated", // 🔥 CRISM 느낌 유지
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

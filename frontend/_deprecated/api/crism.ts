// src/api/crism.ts
export async function fetchCrismRGB(
  productId: string,
  r_um: number,
  g_um: number,
  b_um: number,
  vmin = 0.02,
  vmax = 0.25
): Promise<string> {

  const res = await fetch(`/crism/${productId}/rgb`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      r_um,
      g_um,
      b_um,
      vmin,
      vmax,
    }),
  });

  if (!res.ok) {
    throw new Error("CRISM RGB request failed");
  }

  // 🔑 PNG binary → blob → object URL
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

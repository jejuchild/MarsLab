// src/pages/MastcamLabelPage.tsx
// HiRISE-centric labeling with Mastcam-Z SPICE cross-reference
import { useState, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────
type SolInfo = {
  sol: number;
  product_id: string;
  valid_points: number;
  lon_range: [number, number];
  lat_range: [number, number];
};

type RoughnessClass = "smooth" | "rocky" | "sandy" | "bedrock" | "mixed";
const ROUGHNESS_CLASSES: { id: RoughnessClass; label: string; color: string }[] = [
  { id: "smooth", label: "Smooth", color: "#22c55e" },
  { id: "rocky", label: "Rocky", color: "#ef4444" },
  { id: "sandy", label: "Sandy", color: "#f59e0b" },
  { id: "bedrock", label: "Bedrock", color: "#8b5cf6" },
  { id: "mixed", label: "Mixed", color: "#06b6d4" },
];
const ROCK_SIZE_PRESETS = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 3.0];

type Annotation = {
  pixel_x: number;     // HiRISE grid pixel
  pixel_y: number;
  lon: number;
  lat: number;
  mc_pixel_x: number;  // Mastcam-Z pixel (SPICE mapped)
  mc_pixel_y: number;
  category: string;
  rock_size_m: number;
  confidence: number;
};

// ── Mars projection ───────────────────────────────────
const MARS_R = 3396190.0;
function lonlatToProj(lon: number, lat: number): [number, number] {
  return [(lon / 180) * Math.PI * MARS_R, (lat / 180) * Math.PI * MARS_R];
}
function projToLonlat(x: number, y: number): [number, number] {
  return [(x / (Math.PI * MARS_R)) * 180, (y / (Math.PI * MARS_R)) * 180];
}

// ── HiRISE Labeling Canvas (main interaction) ─────────
function HiriseCanvas({
  centerLon,
  centerLat,
  radiusM,
  canvasSize,
  annotations,
  onClickGround,
  cursorLon,
  cursorLat,
  coverageBounds,
}: {
  centerLon: number;
  centerLat: number;
  radiusM: number;
  canvasSize: number;
  annotations: Annotation[];
  onClickGround: (lon: number, lat: number) => void;
  cursorLon: number | null;
  cursorLat: number | null;
  coverageBounds: { lonMin: number; lonMax: number; latMin: number; latMax: number } | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [hover, setHover] = useState<{ lon: number; lat: number } | null>(null);

  function pixelToGround(px: number, py: number): [number, number] {
    const [cx, cy] = lonlatToProj(centerLon, centerLat);
    const scale = canvasSize / (radiusM * 2);
    const mx = cx + (px - canvasSize / 2) / scale;
    const my = cy + (canvasSize / 2 - py) / scale;
    return projToLonlat(mx, my);
  }

  function groundToPixel(lon: number, lat: number): [number, number] {
    const [cx, cy] = lonlatToProj(centerLon, centerLat);
    const [gx, gy] = lonlatToProj(lon, lat);
    const scale = canvasSize / (radiusM * 2);
    return [
      canvasSize / 2 + (gx - cx) * scale,
      canvasSize / 2 - (gy - cy) * scale,
    ];
  }

  function draw() {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = canvasSize;
    canvas.height = canvasSize;
    ctx.drawImage(img, 0, 0, canvasSize, canvasSize);

    // 25cm grid (visible when zoomed in)
    if (radiusM <= 15) {
      const scale = canvasSize / (radiusM * 2);
      const gridStep = 0.25 * scale; // 25cm
      if (gridStep >= 4) {
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 0.5;
        for (let x = gridStep; x < canvasSize; x += gridStep) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvasSize); ctx.stroke();
        }
        for (let y = gridStep; y < canvasSize; y += gridStep) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasSize, y); ctx.stroke();
        }
      }
    }

    // Coverage box (Mastcam-Z SPICE coverage area)
    if (coverageBounds) {
      const [x1, y1] = groundToPixel(coverageBounds.lonMin, coverageBounds.latMax);
      const [x2, y2] = groundToPixel(coverageBounds.lonMax, coverageBounds.latMin);
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.fillStyle = "#22c55e";
      ctx.font = "10px monospace";
      ctx.fillText("Mastcam-Z coverage", x1 + 4, y1 - 4);
    }

    // Annotations
    for (const ann of annotations) {
      const [px, py] = groundToPixel(ann.lon, ann.lat);
      const cls = ROUGHNESS_CLASSES.find((c) => c.id === ann.category);
      ctx.fillStyle = cls?.color || "#fff";
      ctx.globalAlpha = 0.55;
      ctx.fillRect(px - 4, py - 4, 8, 8);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = cls?.color || "#fff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px - 4, py - 4, 8, 8);
    }

    // Cursor (SPICE-mapped point from Mastcam)
    if (cursorLon != null && cursorLat != null) {
      const [px, py] = groundToPixel(cursorLon, cursorLat);
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 13, py); ctx.lineTo(px + 13, py);
      ctx.moveTo(px, py - 13); ctx.lineTo(px, py + 13);
      ctx.stroke();
    }

    // Hover
    if (hover) {
      const [px, py] = groundToPixel(hover.lon, hover.lat);
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Scale bar
    const barM = radiusM > 20 ? 10 : radiusM > 5 ? 2 : 1;
    const barPx = barM * (canvasSize / (radiusM * 2));
    ctx.fillStyle = "#fff";
    ctx.fillRect(10, canvasSize - 20, barPx, 3);
    ctx.font = "11px monospace";
    ctx.fillText(`${barM}m`, 10, canvasSize - 24);

    // North arrow
    ctx.fillStyle = "#fff";
    ctx.font = "bold 12px monospace";
    ctx.fillText("N", canvasSize - 20, 18);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(canvasSize - 14, 22);
    ctx.lineTo(canvasSize - 14, 38);
    ctx.stroke();
  }

  // Load HiRISE tile
  useEffect(() => {
    const url = `/api/mastcam-label/hirise-tile?lon=${centerLon}&lat=${centerLat}&radius_m=${radiusM}&width=${canvasSize}&height=${canvasSize}`;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
    img.src = url;
  }, [centerLon, centerLat, radiusM, canvasSize]);

  useEffect(() => { draw(); }, [cursorLon, cursorLat, annotations, hover]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const [lon, lat] = pixelToGround(px, py);
    onClickGround(lon, lat);
  }

  function handleMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const [lon, lat] = pixelToGround(px, py);
    setHover({ lon, lat });
  }

  // Arrow key navigation: move 25cm (1 HiRISE pixel) per press
  function handleKeyDown(e: React.KeyboardEvent) {
    if (!cursorLon || !cursorLat) return;
    const step_m = e.shiftKey ? 1.25 : 0.25; // Shift = 5 pixels
    const step_deg_lon = step_m / (59274.0 * Math.cos(cursorLat * Math.PI / 180));
    const step_deg_lat = step_m / 59274.0;
    let newLon = cursorLon, newLat = cursorLat;
    switch (e.key) {
      case "ArrowUp":    newLat += step_deg_lat; break;
      case "ArrowDown":  newLat -= step_deg_lat; break;
      case "ArrowRight": newLon += step_deg_lon; break;
      case "ArrowLeft":  newLon -= step_deg_lon; break;
      default: return;
    }
    e.preventDefault();
    onClickGround(newLon, newLat);
  }

  return (
    <canvas
      ref={canvasRef}
      width={canvasSize}
      height={canvasSize}
      className="border border-[#2d3a54] rounded cursor-crosshair outline-none"
      tabIndex={0}
      onClick={handleClick}
      onMouseMove={handleMove}
      onMouseLeave={() => setHover(null)}
      onKeyDown={handleKeyDown}
    />
  );
}

// ── Mastcam Full Image with Accurate Marker ───────────
function MastcamFullWithMarker({
  sol,
  markerX,
  markerY,
}: {
  sol: number;
  markerX: number | null;
  markerY: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgNat, setImgNat] = useState({ w: 1648, h: 1200 });

  function draw() {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Fit into 400x180 container
    const maxW = 396, maxH = 176;
    const aspect = imgNat.w / imgNat.h;
    let dw = maxW, dh = maxW / aspect;
    if (dh > maxH) { dh = maxH; dw = maxH * aspect; }
    dw = Math.floor(dw); dh = Math.floor(dh);

    canvas.width = dw;
    canvas.height = dh;
    ctx.drawImage(img, 0, 0, dw, dh);

    // Draw marker at correct pixel position
    if (markerX != null && markerY != null) {
      const sx = dw / imgNat.w;
      const sy = dh / imgNat.h;
      const px = markerX * sx;
      const py = markerY * sy;

      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 12, py); ctx.lineTo(px + 12, py);
      ctx.moveTo(px, py - 12); ctx.lineTo(px, py + 12);
      ctx.stroke();
    }
  }

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      setImgNat({ w: img.naturalWidth, h: img.naturalHeight });
      draw();
    };
    img.src = `/api/mastcam-spice/texture/${sol}?quality=60`;
  }, [sol]);

  useEffect(() => { draw(); }, [markerX, markerY, imgNat]);

  return (
    <div className="h-48 border-b border-[#1e293b] p-1 flex items-center justify-center bg-black">
      <canvas ref={canvasRef} className="max-w-full max-h-full" />
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────
export default function MastcamLabelPage() {
  const [sols, setSols] = useState<SolInfo[]>([]);
  const [selectedSol, setSelectedSol] = useState<SolInfo | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [roughness, setRoughness] = useState<RoughnessClass>("rocky");
  const [rockSize, setRockSize] = useState(0.1);
  const [viewRadius, setViewRadius] = useState(15);

  // Clicked HiRISE ground point
  const [clickedGround, setClickedGround] = useState<{ lon: number; lat: number } | null>(null);
  // Matched Mastcam-Z pixel(s)
  const [mcMatch, setMcMatch] = useState<{
    pixel_x: number; pixel_y: number;
    pixel_lon: number; pixel_lat: number;
    offset_m: number;
    pixels_in_cell?: number;
    bbox?: { x0: number; y0: number; x1: number; y1: number };
    bbox_size_px?: { w: number; h: number };
  } | null>(null);
  const [cropScale, setCropScale] = useState<{
    scale_m_per_px: number | null;
    crop_width_m: number | null;
  } | null>(null);

  // Load sols
  useEffect(() => {
    fetch("/api/mastcam-spice/sols")
      .then((r) => r.json())
      .then(setSols);
  }, []);

  // Load labels when sol changes
  useEffect(() => {
    if (!selectedSol) return;
    fetch(`/api/mastcam-spice/labels/${selectedSol.sol}`)
      .then((r) => r.json())
      .then((data) => setAnnotations(data.annotations || []));
    setClickedGround(null);
    setMcMatch(null);
  }, [selectedSol]);

  // When HiRISE is clicked: find nearest Mastcam-Z pixel via SPICE
  const handleHiriseClick = useCallback(
    async (lon: number, lat: number) => {
      if (!selectedSol) return;
      setClickedGround({ lon, lat });
      setMcMatch(null);

      const res = await fetch(
        `/api/mastcam-spice/ground-to-pixel/${selectedSol.sol}?lon=${lon}&lat=${lat}`
      );
      const data = await res.json();
      if (data.found) {
        setMcMatch(data);
        fetch(`/api/mastcam-spice/crop-scale/${selectedSol.sol}?x=${data.pixel_x}&y=${data.pixel_y}&radius=96`)
          .then((r) => r.json())
          .then(setCropScale);
      } else {
        setMcMatch(null);
        setCropScale(null);
      }
    },
    [selectedSol]
  );

  // Add annotation
  const addAnnotation = useCallback(() => {
    if (!clickedGround || !mcMatch || !selectedSol) return;
    const ann: Annotation = {
      pixel_x: 0, // HiRISE grid index (could compute from lon/lat)
      pixel_y: 0,
      lon: clickedGround.lon,
      lat: clickedGround.lat,
      mc_pixel_x: mcMatch.pixel_x,
      mc_pixel_y: mcMatch.pixel_y,
      category: roughness,
      rock_size_m: rockSize,
      confidence: 0.8,
    };
    const updated = [...annotations, ann];
    setAnnotations(updated);
    // Auto-save
    fetch("/api/mastcam-spice/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sol: selectedSol.sol,
        product_id: selectedSol.product_id,
        annotations: updated,
      }),
    });
  }, [clickedGround, mcMatch, roughness, rockSize, annotations, selectedSol]);

  const deleteAnnotation = useCallback(
    (idx: number) => {
      if (!selectedSol) return;
      const updated = annotations.filter((_, i) => i !== idx);
      setAnnotations(updated);
      fetch("/api/mastcam-spice/labels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sol: selectedSol.sol,
          product_id: selectedSol.product_id,
          annotations: updated,
        }),
      });
    },
    [annotations, selectedSol]
  );

  const hiriseCenter = selectedSol
    ? {
        lon: (selectedSol.lon_range[0] + selectedSol.lon_range[1]) / 2,
        lat: (selectedSol.lat_range[0] + selectedSol.lat_range[1]) / 2,
      }
    : null;

  return (
    <div className="h-screen flex flex-col bg-[#0a0f18] text-[#e2e8f0]">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-[#1e293b] bg-[#0d1420]/80 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <a href="/" className="text-[#6b7c9c] hover:text-[#c8d6e5] text-sm">&larr; MarsLab</a>
          <h1 className="text-base font-semibold">Mastcam-Z SPICE Labeling</h1>
          <span className="text-xs text-[#4a5a7a]">HiRISE + Mastcam-Z (SPICE co-registered)</span>
        </div>
        {selectedSol && (
          <span className="text-xs text-[#6b7c9c]">
            Sol {selectedSol.sol} &middot; {annotations.length} labels
          </span>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-60 shrink-0 border-r border-[#1e293b] flex flex-col bg-[#0d1420] overflow-y-auto">
          {/* Sol */}
          <div className="p-2 border-b border-[#1e293b]">
            <label className="text-[10px] text-[#6b7c9c] uppercase tracking-wider">Sol</label>
            <select
              className="w-full mt-1 px-2 py-1.5 rounded bg-[#1a2233] border border-[#2d3a54] text-sm text-[#c8d6e5] outline-none"
              value={selectedSol?.sol ?? ""}
              onChange={(e) => {
                const s = sols.find((s) => s.sol === Number(e.target.value));
                setSelectedSol(s || null);
              }}
            >
              <option value="">Select sol...</option>
              {sols.map((s) => (
                <option key={s.sol} value={s.sol}>Sol {s.sol}</option>
              ))}
            </select>
          </div>

          {/* Roughness */}
          <div className="p-2 border-b border-[#1e293b]">
            <label className="text-[10px] text-[#6b7c9c] uppercase tracking-wider">Roughness</label>
            <div className="flex flex-wrap gap-1 mt-1">
              {ROUGHNESS_CLASSES.map((cls) => (
                <button
                  key={cls.id}
                  onClick={() => setRoughness(cls.id)}
                  className={`px-2 py-1 rounded text-xs transition-all ${
                    roughness === cls.id ? "ring-1 ring-white/40 font-semibold" : "opacity-60 hover:opacity-90"
                  }`}
                  style={{ backgroundColor: cls.color + "33", color: cls.color }}
                >{cls.label}</button>
              ))}
            </div>
          </div>

          {/* Rock size */}
          <div className="p-2 border-b border-[#1e293b]">
            <label className="text-[10px] text-[#6b7c9c] uppercase tracking-wider">Rock size</label>
            <div className="flex flex-wrap gap-1 mt-1">
              {ROCK_SIZE_PRESETS.map((s) => (
                <button
                  key={s}
                  onClick={() => setRockSize(s)}
                  className={`px-1.5 py-0.5 rounded text-[10px] ${
                    rockSize === s ? "bg-amber-600 text-white" : "bg-[#1a2233] text-[#6b7c9c]"
                  }`}
                >{s === 0 ? "None" : `${s}m`}</button>
              ))}
            </div>
          </div>

          {/* View radius */}
          <div className="p-2 border-b border-[#1e293b]">
            <label className="text-[10px] text-[#6b7c9c] uppercase tracking-wider">HiRISE radius</label>
            <div className="flex gap-1 mt-1">
              {[5, 15, 30, 50].map((r) => (
                <button
                  key={r}
                  onClick={() => setViewRadius(r)}
                  className={`px-2 py-0.5 rounded text-xs ${
                    viewRadius === r ? "bg-amber-600 text-white" : "bg-[#1a2233] text-[#6b7c9c]"
                  }`}
                >{r}m</button>
              ))}
            </div>
          </div>

          {/* Add label */}
          <div className="p-2 border-b border-[#1e293b]">
            <button
              onClick={addAnnotation}
              disabled={!mcMatch}
              className="w-full py-2 rounded text-sm font-medium bg-amber-600 hover:bg-amber-500 disabled:bg-[#1a2233] disabled:text-[#4a5a7a] transition-colors"
            >Add Label</button>
            {clickedGround && (
              <p className="text-[10px] text-[#6b7c9c] mt-1 text-center">
                {clickedGround.lon.toFixed(6)}&deg;E, {clickedGround.lat.toFixed(6)}&deg;N
                {mcMatch && <><br />MC offset: {mcMatch.offset_m.toFixed(2)}m</>}
              </p>
            )}
          </div>

          {/* Annotations */}
          <div className="flex-1 overflow-y-auto p-2">
            <label className="text-[10px] text-[#6b7c9c] uppercase tracking-wider">
              Labels ({annotations.length})
            </label>
            <div className="space-y-1 mt-1">
              {annotations.map((ann, i) => {
                const cls = ROUGHNESS_CLASSES.find((c) => c.id === ann.category);
                return (
                  <div key={i} className="flex items-center justify-between px-2 py-1 rounded bg-[#1a2233] text-[10px]">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: cls?.color }} />
                      <span className="text-[#c8d6e5] truncate">{ann.lon.toFixed(5)}</span>
                    </div>
                    <button onClick={() => deleteAnnotation(i)} className="text-[#4a5a7a] hover:text-red-400 ml-1">&times;</button>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        {/* Main split view */}
        <main className="flex-1 flex overflow-hidden">
          {!selectedSol ? (
            <div className="flex-1 flex items-center justify-center text-[#4a5a7a]">
              <div className="text-center">
                <p className="text-lg">Select a sol to begin labeling</p>
                <p className="text-xs mt-1">{sols.length} sols available</p>
              </div>
            </div>
          ) : (
            <>
              {/* Left: HiRISE (main labeling canvas) */}
              <div className="flex-1 flex flex-col border-r border-[#1e293b]">
                <div className="px-3 py-1.5 border-b border-[#1e293b] bg-[#0d1420]/60 flex items-center justify-between">
                  <span className="text-xs font-medium">HiRISE (25cm/px) &mdash; Click to label</span>
                  <span className="text-[10px] text-[#4a5a7a]">{viewRadius}m radius</span>
                </div>
                <div className="flex-1 flex items-center justify-center p-2">
                  {hiriseCenter && (
                    <HiriseCanvas
                      centerLon={hiriseCenter.lon}
                      centerLat={hiriseCenter.lat}
                      radiusM={viewRadius}
                      canvasSize={560}
                      annotations={annotations}
                      onClickGround={handleHiriseClick}
                      cursorLon={clickedGround?.lon ?? null}
                      cursorLat={clickedGround?.lat ?? null}
                      coverageBounds={selectedSol ? {
                        lonMin: selectedSol.lon_range[0],
                        lonMax: selectedSol.lon_range[1],
                        latMin: selectedSol.lat_range[0],
                        latMax: selectedSol.lat_range[1],
                      } : null}
                    />
                  )}
                </div>
              </div>

              {/* Right: Mastcam-Z reference */}
              <div className="w-[400px] shrink-0 flex flex-col">
                <div className="px-3 py-1.5 border-b border-[#1e293b] bg-[#0d1420]/60 flex items-center justify-between">
                  <span className="text-xs font-medium">Mastcam-Z Reference</span>
                  <span className="text-[10px] text-[#4a5a7a]">SPICE co-registered</span>
                </div>

                {/* Full image with marker (canvas-based for pixel accuracy) */}
                <MastcamFullWithMarker
                  sol={selectedSol.sol}
                  markerX={mcMatch?.pixel_x ?? null}
                  markerY={mcMatch?.pixel_y ?? null}
                />

                {/* Zoomed crop showing HiRISE cell boundary */}
                <div className="flex-1 flex flex-col items-center justify-center p-2 bg-[#0a0f18]">
                  {mcMatch ? (
                    <>
                      <img
                        src={`/api/mastcam-spice/texture-crop/${selectedSol.sol}?x=${mcMatch.pixel_x}&y=${mcMatch.pixel_y}&radius=96&quality=95${
                          mcMatch.bbox ? `&bbox_x0=${mcMatch.bbox.x0}&bbox_y0=${mcMatch.bbox.y0}&bbox_x1=${mcMatch.bbox.x1}&bbox_y1=${mcMatch.bbox.y1}` : ""
                        }`}
                        alt="Mastcam-Z crop"
                        className="max-w-full border border-[#2d3a54] rounded"
                      />
                      <p className="text-[10px] text-[#6b7c9c] mt-2 text-center">
                        <span className="text-emerald-400">Green box</span> = 1 HiRISE pixel (25cm x 25cm)
                        {mcMatch.pixels_in_cell != null && (
                          <> &middot; {mcMatch.pixels_in_cell} MC-Z pixels inside</>
                        )}
                        {mcMatch.bbox_size_px && (
                          <><br />{mcMatch.bbox_size_px.w} x {mcMatch.bbox_size_px.h} Mastcam-Z pixels</>
                        )}
                        {cropScale?.scale_m_per_px != null && (
                          <><br />{(cropScale.scale_m_per_px * 100).toFixed(2)} cm/px (SPICE-derived)</>
                        )}
                      </p>
                    </>
                  ) : clickedGround ? (
                    <div className="text-center">
                      <p className="text-sm text-red-400 font-medium">No Mastcam-Z coverage</p>
                      <p className="text-[10px] text-[#6b7c9c] mt-1">
                        This HiRISE pixel has no matching Mastcam-Z data.
                        <br />
                        Only 1 of 24 Mastcam-Z frames is loaded for this sol.
                        <br />
                        Coverage is limited to a ~6m x 4m ground area.
                      </p>
                    </div>
                  ) : (
                    <p className="text-xs text-[#4a5a7a]">
                      Click on HiRISE to see Mastcam-Z view
                    </p>
                  )}
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

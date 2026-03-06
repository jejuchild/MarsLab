import { useEffect } from "react";
import type React from "react";
import * as Cesium from "cesium";
import { LRUMap } from "../utils/LRUMap";
import type { FieldNote } from "../api/fieldnotes";

// Field note marker icon colors by instrument
const FIELDNOTE_COLORS: Record<string, string> = {
  CRISM: "#22d3ee",      // cyan
  HIRISE: "#facc15",     // yellow
  SHARAD: "#fb923c",     // orange
  SHARAD_HIGHRES: "#fb923c",
  CTX: "#f472b6",        // pink
  CUSTOM: "#e879f9",     // fuchsia
  HIRISE_DTM: "#d97706", // amber
};

// Create a canvas-based icon for field note marker (cached per instrument)
const _fieldNoteIconCache = new LRUMap<string, string>(64);
function createFieldNoteIcon(instrument: string): string {
  const cached = _fieldNoteIconCache.get(instrument);
  if (cached) return cached;

  const canvas = document.createElement("canvas");
  canvas.width = 24;
  canvas.height = 24;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const color = FIELDNOTE_COLORS[instrument] || "#fbbf24";

  // Draw pin shape
  ctx.beginPath();
  ctx.arc(12, 9, 7, Math.PI, 0, false);
  ctx.lineTo(12, 22);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Inner circle (white)
  ctx.beginPath();
  ctx.arc(12, 9, 3, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();

  const dataUrl = canvas.toDataURL();
  _fieldNoteIconCache.set(instrument, dataUrl);
  return dataUrl;
}

type UseAnnotationsParams = {
  viewerRef: React.MutableRefObject<Cesium.Viewer | null>;
  marsEllipsoid: Cesium.Ellipsoid;
  analysisMode: string | null;
  linePoints: Array<{ lat: number; lon: number }>;
  fieldNotes: FieldNote[];
  aiAnalysisPin: { lat: number; lon: number } | null;
  sharadTracePin: { lat: number; lon: number } | null;
  terraformMode: boolean;
  craterDetectFeatures?: Array<{
    id: string;
    type: string;
    lat: number;
    lon: number;
    diameter_km?: number;
    area_km2?: number;
    length_km?: number;
    morphology?: string;
    confidence: number;
    description: string;
    path?: [number, number][];
    boundary?: [number, number][];
  }>;
};

export default function useAnnotations({
  viewerRef,
  marsEllipsoid,
  analysisMode,
  linePoints,
  fieldNotes,
  aiAnalysisPin,
  sharadTracePin,
  terraformMode,
  craterDetectFeatures,
}: UseAnnotationsParams): void {

  // Line Profile markers and polyline
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Entity IDs for line profile
    const LP_MARKER_A = "LINE_PROFILE_MARKER_A";
    const LP_MARKER_B = "LINE_PROFILE_MARKER_B";
    const LP_LABEL_A = "LINE_PROFILE_LABEL_A";
    const LP_LABEL_B = "LINE_PROFILE_LABEL_B";
    const LP_LINE = "LINE_PROFILE_LINE";

    // Clear all line profile entities
    const clearAll = () => {
      for (const id of [LP_MARKER_A, LP_MARKER_B, LP_LABEL_A, LP_LABEL_B, LP_LINE]) {
        const ent = viewer.entities.getById(id);
        if (ent) viewer.entities.remove(ent);
      }
    };

    clearAll();

    if (analysisMode !== "line" || linePoints.length === 0) {
      viewer.scene.requestRender();
      return;
    }

    const fmtLabel = (lat: number, lon: number) =>
      `${Math.abs(lat).toFixed(4)}\u00b0${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}\u00b0${lon >= 0 ? "E" : "W"}`;

    // First point marker
    const p1 = linePoints[0]!;
    viewer.entities.add({
      id: LP_MARKER_A,
      position: Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, 0, marsEllipsoid),
      point: {
        pixelSize: 8,
        color: Cesium.Color.LIME,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    viewer.entities.add({
      id: LP_LABEL_A,
      position: Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, 0, marsEllipsoid),
      label: {
        text: fmtLabel(p1.lat, p1.lon),
        font: "11px monospace",
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -12),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // Second point + polyline
    if (linePoints.length >= 2) {
      const p2 = linePoints[1]!;
      viewer.entities.add({
        id: LP_MARKER_B,
        position: Cesium.Cartesian3.fromDegrees(p2.lon, p2.lat, 0, marsEllipsoid),
        point: {
          pixelSize: 8,
          color: Cesium.Color.RED,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      viewer.entities.add({
        id: LP_LABEL_B,
        position: Cesium.Cartesian3.fromDegrees(p2.lon, p2.lat, 0, marsEllipsoid),
        label: {
          text: fmtLabel(p2.lat, p2.lon),
          font: "11px monospace",
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -12),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
      viewer.entities.add({
        id: LP_LINE,
        polyline: {
          positions: [
            Cesium.Cartesian3.fromDegrees(p1.lon, p1.lat, 0, marsEllipsoid),
            Cesium.Cartesian3.fromDegrees(p2.lon, p2.lat, 0, marsEllipsoid),
          ],
          width: 2,
          material: Cesium.Color.LIME.withAlpha(0.8),
          clampToGround: true,
        },
      });
    }

    viewer.scene.requestRender();
  }, [analysisMode, linePoints, viewerRef, marsEllipsoid]);

  // Crater/Landform detection entities
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // Remove old detection entities
    const toRemove = viewer.entities.values.filter(
      (e: Cesium.Entity) => e.id?.startsWith("DETECT_")
    );
    for (const e of toRemove) viewer.entities.remove(e);

    if (!craterDetectFeatures || craterDetectFeatures.length === 0) {
      viewer.scene.requestRender();
      return;
    }

    const COLORS: Record<string, string> = {
      crater: "#fb923c",
      terraced_crater: "#f43f5e",
      volcanic: "#ef4444",
      graben: "#a855f7",
      channel: "#3b82f6",
      wrinkle_ridge: "#eab308",
      lda: "#22d3ee",
    };

    viewer.entities.suspendEvents();

    for (const f of craterDetectFeatures) {
      const color = COLORS[f.type] || "#6b7c9c";
      const cesiumColor = Cesium.Color.fromCssColorString(color);

      if (f.type === "lda" && f.boundary && f.boundary.length > 2) {
        // LDA: polygon
        const positions = f.boundary.map(([lat, lon]: [number, number]) =>
          Cesium.Cartesian3.fromDegrees(lon, lat, 0, marsEllipsoid)
        );
        viewer.entities.add({
          id: `DETECT_${f.id}`,
          polygon: {
            hierarchy: positions,
            material: cesiumColor.withAlpha(0.25),
            outline: true,
            outlineColor: cesiumColor.withAlpha(0.8),
          },
        });
        viewer.entities.add({
          id: `DETECT_L_${f.id}`,
          position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat, 0, marsEllipsoid),
          label: {
            text: `LDA ${f.area_km2?.toFixed(0) || ""} km\u00b2`,
            font: "bold 11px sans-serif",
            fillColor: cesiumColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 2e6, 0.3),
          },
        });
      } else if ((f.type === "channel" || f.type === "wrinkle_ridge" || f.type === "graben") && f.path && f.path.length > 1) {
        // Polyline features
        const positions = f.path.map(([lat, lon]: [number, number]) =>
          Cesium.Cartesian3.fromDegrees(lon, lat, 0, marsEllipsoid)
        );
        viewer.entities.add({
          id: `DETECT_${f.id}`,
          polyline: {
            positions,
            width: f.type === "graben" ? 3 : 2,
            material: cesiumColor.withAlpha(0.8),
            clampToGround: true,
          },
        });
        const midIdx = Math.floor(f.path.length / 2);
        const midPt = f.path[midIdx];
        if (!midPt) continue;
        const sizeLabel = f.length_km ? `${f.length_km.toFixed(0)} km` : "";
        viewer.entities.add({
          id: `DETECT_L_${f.id}`,
          position: Cesium.Cartesian3.fromDegrees(midPt[1], midPt[0], 0, marsEllipsoid),
          label: {
            text: `${f.type === "channel" ? "Ch" : f.type === "graben" ? "Gr" : "WR"} ${sizeLabel}`,
            font: "10px sans-serif",
            fillColor: cesiumColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 2e6, 0.3),
          },
        });
      } else {
        // Craters, volcanics — ellipse + visible point marker
        const radiusM = (f.diameter_km || 5) * 500;
        viewer.entities.add({
          id: `DETECT_${f.id}`,
          position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat, 0, marsEllipsoid),
          ellipse: {
            semiMajorAxis: radiusM,
            semiMinorAxis: radiusM,
            material: cesiumColor.withAlpha(0.2),
            outline: true,
            outlineColor: cesiumColor.withAlpha(0.8),
          },
          point: {
            pixelSize: 7,
            color: cesiumColor,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 1,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label: {
            text: `${f.morphology || f.type}\n${f.diameter_km?.toFixed(1) || ""} km`,
            font: "10px sans-serif",
            fillColor: cesiumColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE,
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            pixelOffset: new Cesium.Cartesian2(0, -12),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            scaleByDistance: new Cesium.NearFarScalar(5e4, 1.0, 2e6, 0.3),
          },
        });
      }
    }

    viewer.entities.resumeEvents();
    viewer.scene.requestRender();

    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      const ents = viewer.entities.values.filter(
        (e: Cesium.Entity) => e.id?.startsWith("DETECT_")
      );
      for (const e of ents) viewer.entities.remove(e);
    };
  }, [craterDetectFeatures, viewerRef, marsEllipsoid]);

  // Easter egg: Terraform mode — tint globe blue/green
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const TERRAFORM_ID = "__TERRAFORM_TINT__";

    if (terraformMode) {
      viewer.entities.add({
        id: TERRAFORM_ID,
        rectangle: {
          coordinates: Cesium.Rectangle.fromDegrees(-180, -90, 180, 90),
          material: new Cesium.ColorMaterialProperty(
            new Cesium.Color(0.1, 0.5, 0.8, 0.3)
          ),
          height: 0,
        },
      });
      viewer.scene.requestRender();
    } else {
      const ent = viewer.entities.getById(TERRAFORM_ID);
      if (ent) {
        viewer.entities.remove(ent);
        viewer.scene.requestRender();
      }
    }

    return () => {
      if (!viewer || viewer.isDestroyed()) return;
      const ent = viewer.entities.getById(TERRAFORM_ID);
      if (ent) viewer.entities.remove(ent);
    };
  }, [terraformMode, viewerRef]);

  // Field Note indicators (independent of layer state)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const PREFIX = "FIELDNOTE_";
    let cancelled = false;

    // Helper to add a field note marker
    const addFieldNoteMarker = (
      v: Cesium.Viewer,
      note: { id: string; product_id: string; instrument: string },
      lat: number,
      lon: number
    ) => {
      // Check if already exists
      if (v.entities.getById(`${PREFIX}${note.id}`)) return;

      v.entities.add({
        id: `${PREFIX}${note.id}`,
        position: Cesium.Cartesian3.fromDegrees(lon, lat, 0),
        billboard: {
          image: createFieldNoteIcon(note.instrument),
          width: 24,
          height: 24,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: "\u2605", // star character
          font: "12px sans-serif",
          fillColor: Cesium.Color.GOLD,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          pixelOffset: new Cesium.Cartesian2(0, -28),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        properties: {
          type: "fieldnote",
          noteId: note.id,
          productId: note.product_id,
          instrument: note.instrument,
        },
      });
    };

    viewer.entities.suspendEvents();
    try {
      // Remove existing field note markers
      const toRemove: Cesium.Entity[] = [];
      for (let i = 0; i < viewer.entities.values.length; i++) {
        const ent = viewer.entities.values[i]!;
        if (ent.id.startsWith(PREFIX)) toRemove.push(ent);
      }
      for (const ent of toRemove) viewer.entities.remove(ent);

      if (!fieldNotes || fieldNotes.length === 0) {
        return;
      }

      // Collect notes that need coordinate lookup
      const notesNeedingCoords: typeof fieldNotes = [];

      // Create markers at each field note's lat/lon (works regardless of layer state)
      for (const note of fieldNotes) {
        const lat = note.lat;
        const lon = note.lon;

        // If coordinates are 0,0, we need to look them up
        if (lat === 0 && lon === 0) {
          notesNeedingCoords.push(note);
          continue;
        }

        addFieldNoteMarker(viewer, note, lat, lon);
      }

      // Fetch coordinates for notes with 0,0 coords (async)
      if (notesNeedingCoords.length > 0) {
        // Group by instrument to minimize API calls
        const byInstrument = new Map<string, typeof fieldNotes>();
        for (const note of notesNeedingCoords) {
          const inst = note.instrument;
          if (!byInstrument.has(inst)) byInstrument.set(inst, []);
          byInstrument.get(inst)!.push(note);
        }

        // Fetch footprints for each instrument
        for (const [instrument, notes] of byInstrument) {
          (async () => {
            try {
              const res = await fetch(
                `/api/footprints?instrument=${instrument}&bbox=-180,-90,180,90&limit=5000&lod=poly`
              );
              if (!res.ok || cancelled) return;

              const data = await res.json();
              if (cancelled) return;

              const v = viewerRef.current;
              if (!v || v.isDestroyed()) return;

              v.entities.suspendEvents();
              try {
                for (const note of notes) {
                  const feat = data.features?.find(
                    (f: any) => f.properties?.product_id === note.product_id
                  );
                  if (!feat?.geometry?.coordinates) continue;

                  let lat = 0, lon = 0;
                  const geom = feat.geometry;

                  if (geom.type === "LineString" && geom.coordinates.length >= 2) {
                    // For LineStrings (SHARAD tracks), use midpoint
                    const midIdx = Math.floor(geom.coordinates.length / 2);
                    lon = geom.coordinates[midIdx][0];
                    lat = geom.coordinates[midIdx][1];
                  } else if (geom.type === "Polygon" && geom.coordinates[0]?.length > 0) {
                    // For Polygons, compute centroid
                    const ring = geom.coordinates[0];
                    let sumLat = 0, sumLon = 0;
                    for (const [plon, plat] of ring) {
                      sumLon += plon;
                      sumLat += plat;
                    }
                    lon = sumLon / ring.length;
                    lat = sumLat / ring.length;
                  } else if (geom.type === "Point") {
                    lon = geom.coordinates[0];
                    lat = geom.coordinates[1];
                  }

                  if (lat !== 0 || lon !== 0) {
                    addFieldNoteMarker(v, note, lat, lon);
                  }
                }
              } finally {
                v.entities.resumeEvents();
              }
              v.scene.requestRender();
            } catch (err) {
              console.error(`[FieldNotes] Failed to fetch coords for ${instrument}:`, err);
            }
          })();
        }
      }
    } finally {
      viewer.entities.resumeEvents();
    }

    viewer.scene.requestRender();

    return () => {
      cancelled = true;
    };
  }, [fieldNotes, viewerRef]);

  // AI Analysis Pin + Radius Circle
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const PIN_ID = "AI_ANALYSIS_PIN";
    const RADIUS_ID = "AI_ANALYSIS_RADIUS";

    // Remove existing entities
    const oldPin = viewer.entities.getById(PIN_ID);
    if (oldPin) viewer.entities.remove(oldPin);
    const oldRadius = viewer.entities.getById(RADIUS_ID);
    if (oldRadius) viewer.entities.remove(oldRadius);

    if (!aiAnalysisPin) {
      viewer.scene.requestRender();
      return;
    }

    const { lat, lon } = aiAnalysisPin;

    // Pin point
    viewer.entities.add({
      id: PIN_ID,
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, marsEllipsoid),
      point: {
        pixelSize: 10,
        color: Cesium.Color.fromCssColorString("#8b5cf6"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `AI Analysis\n${lat.toFixed(3)}°, ${lon.toFixed(3)}°`,
        font: "11px sans-serif",
        fillColor: Cesium.Color.fromCssColorString("#8b5cf6"),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -14),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    // Radius circle (10 km default; visual only)
    viewer.entities.add({
      id: RADIUS_ID,
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, marsEllipsoid),
      ellipse: {
        semiMajorAxis: 10000,  // 10 km
        semiMinorAxis: 10000,
        material: Cesium.Color.fromCssColorString("#8b5cf6").withAlpha(0.12),
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString("#8b5cf6").withAlpha(0.5),
        outlineWidth: 1,
      },
    });

    viewer.scene.requestRender();
  }, [aiAnalysisPin, viewerRef, marsEllipsoid]);

  // Clean up AI Analysis entities when mode changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (analysisMode !== "ai_analysis") {
      const pin = viewer.entities.getById("AI_ANALYSIS_PIN");
      if (pin) viewer.entities.remove(pin);
      const radius = viewer.entities.getById("AI_ANALYSIS_RADIUS");
      if (radius) viewer.entities.remove(radius);
      viewer.scene.requestRender();
    }
  }, [analysisMode, viewerRef]);

  // SHARAD Trace Pin
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const TRACE_PIN_ID = "SHARAD_TRACE_PIN";
    const old = viewer.entities.getById(TRACE_PIN_ID);
    if (old) viewer.entities.remove(old);

    if (!sharadTracePin) {
      viewer.scene.requestRender();
      return;
    }

    const { lat, lon } = sharadTracePin;
    viewer.entities.add({
      id: TRACE_PIN_ID,
      position: Cesium.Cartesian3.fromDegrees(lon, lat, 0, marsEllipsoid),
      point: {
        pixelSize: 10,
        color: Cesium.Color.fromCssColorString("#f59e0b"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `${lat.toFixed(3)}°, ${lon.toFixed(3)}°`,
        font: "bold 11px monospace",
        fillColor: Cesium.Color.fromCssColorString("#f59e0b"),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -12),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });

    viewer.scene.requestRender();
  }, [sharadTracePin, viewerRef, marsEllipsoid]);
}

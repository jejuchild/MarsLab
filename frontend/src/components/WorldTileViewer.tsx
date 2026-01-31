// src/WorldTileViewer.tsx
import { useEffect, useRef, useState } from "react";
import "ol/ol.css";
import Map from "ol/Map";
import View from "ol/View";
import Projection from "ol/proj/Projection";
import TileLayer from "ol/layer/Tile";
import XYZ from "ol/source/XYZ";
import TileGrid from "ol/tilegrid/TileGrid";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import Feature from "ol/Feature";
import Point from "ol/geom/Point";
import { Style, Circle as CircleStyle, Fill, Stroke } from "ol/style";

type Props = {
  productId: string;

  pin?: { x: number; y: number } | null;

  onDoubleClick?: (xy: { x: number; y: number }) => void;

  // 🟢 ADD: viewport extent 전달
  onViewExtentChange?: (extent: [number, number, number, number]) => void;
};

const TILE_SIZE = 256;
const BASE_MAX_ZOOM = 8;
const EXTRA_ZOOM = 2;
const MIN_ZOOM = 0;

export default function WorldTileViewer({
  productId,
  pin,
  onDoubleClick,
  onViewExtentChange,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const layerRef = useRef<TileLayer<XYZ> | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const pinLayerRef = useRef<VectorLayer<VectorSource> | null>(null);

  const [localPin, setLocalPin] = useState<{ x: number; y: number } | null>(
    null
  );

  const effectivePin = pin ?? localPin;

  /* ======================================================
   * Create map
   * ======================================================*/
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let destroyed = false;

    async function createMap() {
      if (mapRef.current) return;
      if (el.clientWidth === 0 || el.clientHeight === 0) return;

      const metaRes = await fetch("/world_meta");
      if (!metaRes.ok || destroyed) return;

      const meta = await metaRes.json();
      const extent = meta.extent as [number, number, number, number];

      const resolutions: number[] = [];
      for (let z = MIN_ZOOM; z <= BASE_MAX_ZOOM + EXTRA_ZOOM; z++) {
        resolutions.push(Math.pow(2, BASE_MAX_ZOOM - z));
      }

      const projection = new Projection({
        code: "MARS_WORLD",
        units: "m",
        extent,
      });

      const tileGrid = new TileGrid({
        origin: [extent[0], extent[3]],
        tileSize: TILE_SIZE,
        resolutions,
      });

      const source = new XYZ({
        url: `/world_tiles/${productId}/{z}/{x}/{y}.png`,
        projection,
        tileGrid,
        wrapX: false,
        crossOrigin: "anonymous",
      });

      const layer = new TileLayer({ source });

      const pinSource = new VectorSource();
      const pinLayer = new VectorLayer({
        source: pinSource,
        style: new Style({
          image: new CircleStyle({
            radius: 6,
            fill: new Fill({ color: "rgba(255,0,0,0.9)" }),
            stroke: new Stroke({ color: "#fff", width: 2 }),
          }),
        }),
      });

      const map = new Map({
        target: el,
        layers: [layer, pinLayer],
        view: new View({
          projection,
          minZoom: MIN_ZOOM,
          maxZoom: BASE_MAX_ZOOM + EXTRA_ZOOM,
          constrainResolution: false,
        }),
      });

      const view = map.getView();
      view.fit(extent, { constrainResolution: false });
      map.updateSize();

      // 🟢 ADD: view extent emit 함수
      const emitExtent = () => {
        if (!mapRef.current) return;
        const size = map.getSize();
        if (!size) return;
        const e = view.calculateExtent(size) as [
          number,
          number,
          number,
          number
        ];
        onViewExtentChange?.(e);
      };

      // 최초 1회
      emitExtent();

      // 이동 / 줌 변화 감지
      view.on("change:center", emitExtent);
      view.on("change:resolution", emitExtent);

      const togglePin = (coord: [number, number]) => {
        const [x, y] = coord;
        setLocalPin((prev) => (prev ? null : { x, y }));
        onDoubleClick?.({ x, y });
      };

      map.on("singleclick", (evt) => {
        togglePin(evt.coordinate as [number, number]);
      });

      map.on("dblclick", (evt) => {
        evt.preventDefault();
        togglePin(evt.coordinate as [number, number]);
      });

      mapRef.current = map;
      layerRef.current = layer;
      pinLayerRef.current = pinLayer;
    }

    resizeObserverRef.current = new ResizeObserver(() => {
      createMap();
      mapRef.current?.updateSize();
    });

    resizeObserverRef.current.observe(el);
    createMap();

    return () => {
      destroyed = true;

      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;

      if (mapRef.current) {
        mapRef.current.setTarget(undefined);
        mapRef.current = null;
      }

      layerRef.current = null;
      pinLayerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ======================================================
   * productId 변경 → 타일 URL만 교체
   * ======================================================*/
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;

    const src = layer.getSource();
    if (!src) return;

    src.setUrl(
      `/world_tiles/${productId}/{z}/{x}/{y}.png`
    );
    src.refresh();
  }, [productId]);

  /* ======================================================
   * pin 렌더링
   * ======================================================*/
  useEffect(() => {
    const pinLayer = pinLayerRef.current;
    if (!pinLayer) return;

    const src = pinLayer.getSource();
    if (!src) return;

    src.clear();

    if (!effectivePin) return;

    const feature = new Feature({
      geometry: new Point([effectivePin.x, effectivePin.y]),
    });

    src.addFeature(feature);
    mapRef.current?.render();
  }, [effectivePin]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        background: "#000",
      }}
    />
  );
}

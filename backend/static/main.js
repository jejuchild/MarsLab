const images = [
  "ESP_024943_2345_RED",
  "ESP_045290_2350_RED_025m",
  "ESP_045857_2350_RED"
];

let map, layer;
let projection, tileGrid;
let activeIndex = 0;

// ----------------------------
// Config
// ----------------------------
const TILE_SIZE = 256;
const BASE_MAX_ZOOM = 8;
const EXTRA_ZOOM = 2;
const MIN_ZOOM = 0;

// ----------------------------
// Fetch world meta
// ----------------------------
async function fetchWorldMeta() {
  const res = await fetch(`http://localhost:8000/world_meta`);
  return await res.json();
}

function makeResolutions() {
  const resolutions = [];
  for (let z = MIN_ZOOM; z <= BASE_MAX_ZOOM + EXTRA_ZOOM; z++) {
    resolutions.push(Math.pow(2, BASE_MAX_ZOOM - z));
  }
  return resolutions;
}

// ----------------------------
// Init
// ----------------------------
async function init() {
  const wm = await fetchWorldMeta();
  const extent = wm.extent; // [minX, minY, maxX, maxY] in "world units"
  const resolutions = makeResolutions();

  // world projection (units=meters-ish; 실제 CRS는 몰라도 됨, 연속 좌표면 충분)
  projection = new ol.proj.Projection({
    code: 'MARS_WORLD',
    units: 'm',
    extent: extent
  });

    tileGrid = new ol.tilegrid.TileGrid({
    origin: [extent[0], extent[3]], // 🔥 좌상단 (minX, maxY)
    tileSize: TILE_SIZE,
    resolutions: resolutions
    });

  // 한 번만 source 생성
  const source = new ol.source.XYZ({
    url: `http://localhost:8000/world_tiles/${images[0]}/{z}/{x}/{y}.png`,
    projection: projection,
    tileGrid: tileGrid,
    wrapX: false,
    crossOrigin: 'anonymous'
  });

  layer = new ol.layer.Tile({ source });

  map = new ol.Map({
    target: 'map',
    layers: [layer],
    view: new ol.View({
      projection: projection,
      minZoom: MIN_ZOOM,
      maxZoom: BASE_MAX_ZOOM + EXTRA_ZOOM,
      constrainResolution: false
    })
  });

  // QGIS 스타일로 전체 월드 extent fit
  map.getView().fit(extent, {
    size: map.getSize(),
    constrainResolution: false
  });

  buildPanel();
}

// ----------------------------
// Switch image (source 재생성 X, URL만 교체)
// ----------------------------
function switchImage(index) {
  const name = images[index];
  const src = layer.getSource();

  // 캐시/타일 상태를 최대한 유지하며 URL만 변경
  if (src && typeof src.setUrl === "function") {
    src.setUrl(`http://localhost:8000/world_tiles/${name}/{z}/{x}/{y}.png`);
    // URL 바꾸면 타일 캐시 갱신 트리거
    src.refresh();
  }

  document.querySelectorAll("#layer-panel div").forEach((d, i) =>
    d.classList.toggle("active", i === index)
  );

  activeIndex = index;
}

// ----------------------------
// UI panel
// ----------------------------
function buildPanel() {
  const panel = document.getElementById("layer-panel");
  panel.innerHTML = "";

  images.forEach((name, i) => {
    const div = document.createElement("div");
    div.textContent = name;
    if (i === 0) div.classList.add("active");
    div.onclick = () => switchImage(i);
    panel.appendChild(div);
  });
}

// ----------------------------
// Keyboard support
// ----------------------------
window.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") {
    switchImage((activeIndex + 1) % images.length);
  }
  if (e.key === "ArrowLeft") {
    switchImage((activeIndex - 1 + images.length) % images.length);
  }
});

init();

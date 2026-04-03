import * as Cesium from "cesium";

const TILE_SIZE = 256;
const EPSILON = 1e-9;

const MARS_ELLIPSOID = new Cesium.Ellipsoid(3396190, 3396190, 3376200);

function getSpacingForLevel(level: number): number {
  if (level <= 2) return 30;
  if (level <= 4) return 10;
  if (level <= 6) return 5;
  if (level <= 8) return 1;
  return 0.5;
}

export class MarsGridImageryProvider {
  private _tilingScheme: Cesium.GeographicTilingScheme;
  private _tileWidth = TILE_SIZE;
  private _tileHeight = TILE_SIZE;
  private _errorEvent = new Cesium.Event();

  constructor() {
    this._tilingScheme = new Cesium.GeographicTilingScheme({
      ellipsoid: MARS_ELLIPSOID,
    });
  }

  get tileWidth(): number {
    return this._tileWidth;
  }

  get tileHeight(): number {
    return this._tileHeight;
  }

  get tilingScheme(): Cesium.GeographicTilingScheme {
    return this._tilingScheme;
  }

  get errorEvent(): Cesium.Event {
    return this._errorEvent;
  }

  get rectangle(): Cesium.Rectangle {
    return this._tilingScheme.rectangle;
  }

  get maximumLevel(): number {
    return 12;
  }

  get minimumLevel(): number {
    return 0;
  }

  get credit(): Cesium.Credit | undefined {
    return undefined;
  }

  get ready(): boolean {
    return true;
  }

  get hasAlphaChannel(): boolean {
    return true;
  }

  getTileCredits(): Cesium.Credit[] {
    return [];
  }

  requestImage(x: number, y: number, level: number): Promise<HTMLCanvasElement> | undefined {
    const canvas = document.createElement("canvas");
    canvas.width = this._tileWidth;
    canvas.height = this._tileHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;

    const rect = this._tilingScheme.tileXYToRectangle(x, y, level);
    const west = Cesium.Math.toDegrees(rect.west);
    const south = Cesium.Math.toDegrees(rect.south);
    const east = Cesium.Math.toDegrees(rect.east);
    const north = Cesium.Math.toDegrees(rect.north);

    const spacing = getSpacingForLevel(level);

    const degToPixelX = (deg: number) => ((deg - west) / (east - west)) * this._tileWidth;
    const degToPixelY = (deg: number) => ((north - deg) / (north - south)) * this._tileHeight;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;

    const startLat = Math.ceil((south - EPSILON) / spacing) * spacing;
    for (let lat = startLat; lat <= north + EPSILON; lat += spacing) {
      const py = degToPixelY(lat);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(this._tileWidth, py);
      ctx.stroke();
    }

    const startLon = Math.ceil((west - EPSILON) / spacing) * spacing;
    for (let lon = startLon; lon <= east + EPSILON; lon += spacing) {
      const px = degToPixelX(lon);
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, this._tileHeight);
      ctx.stroke();
    }

    if (level >= 3) {
      ctx.font = "11px monospace";
      ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
      ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
      ctx.lineWidth = 2;
      ctx.textBaseline = "bottom";

      for (let lat = startLat; lat <= north + EPSILON; lat += spacing) {
        const py = degToPixelY(lat);
        for (let lon = startLon; lon <= east + EPSILON; lon += spacing) {
          const px = degToPixelX(lon);
          if (px > 5 && px < this._tileWidth - 30 && py > 12 && py < this._tileHeight - 5) {
            const label = `${lat.toFixed(spacing < 1 ? 1 : 0)}°`;
            ctx.strokeText(label, px + 3, py - 2);
            ctx.fillText(label, px + 3, py - 2);
          }
        }
      }
    }

    return Promise.resolve(canvas);
  }
}

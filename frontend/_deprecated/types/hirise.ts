export type HiRISEWindowStats = {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  count: number;
};

export type HiRISEHistogram = {
  bins: number;
  counts: number[];
};

export type HiRISEWindowResponse = {
  productId: string;
  projected_xy: {
    x: number;
    y: number;
  };
  pixel_center: {
    row: number;
    col: number;
  };
  window: {
    row_range: [number, number];
    col_range: [number, number];
    size: [number, number];
  };
  stats: HiRISEWindowStats;
  histogram: HiRISEHistogram;
};

// detail-inspector/panels/hirise/HiRISEHistogramPanel.tsx
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

/* =========================================================
 * Props
 * =======================================================*/
type Props = {
  counts?: number[];     // 🔥 optional
  centerBin?: number;
};

/* =========================================================
 * HiRISE Histogram Panel
 * =======================================================*/
export default function HiRISEHistogramPanel({
  counts,
  centerBin,
}: Props) {
  // 🔒 SAFETY GUARD
  if (!counts || counts.length === 0) {
    return (
      <div
        style={{
          fontSize: 12,
          opacity: 0.6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
        }}
      >
        Histogram not available
      </div>
    );
  }

  const data = counts.map((v, i) => ({
    bin: i,
    count: v,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <XAxis dataKey="bin" hide />

        <YAxis
          yAxisId="count"
          tick={{ fontSize: 10 }}
          domain={[0, "dataMax"]}
        />

        <Tooltip />

        {centerBin !== undefined && (
          <ReferenceLine
            x={centerBin}
            yAxisId="count"
            stroke="#ff0000"
            strokeWidth={3}
            strokeDasharray="6 4"
            isFront
          />
        )}

        <Bar
          yAxisId="count"
          dataKey="count"
          fill="#8884d8"
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

import type {
  HiRISEWindowStats,
  HiRISEWindowCenter,
} from "../../../../types/hirise";

/* =========================================================
 * Props
 * =======================================================*/
type Props = {
  stats: HiRISEWindowStats;
  center?: HiRISEWindowCenter;
};

/* =========================================================
 * HiRISE Stats Panel
 * (former StatsTable, logic unchanged)
 * =======================================================*/
export default function HiRISEStatsPanel({ stats, center }: Props) {
  const z =
    center?.zscore !== null && center?.zscore !== undefined
      ? center.zscore
      : null;

  return (
    <div
      style={{
        height: "100%",
        minHeight: 0,
        overflowY: "auto",
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 12,
        }}
      >
        <tbody>
          {/* ===== Center pixel ===== */}
          {center && (
            <>
              <tr>
                <td style={{ opacity: 0.7 }}>Center DN</td>
                <td>
                  <b>{center.dn}</b>
                </td>
              </tr>
              <tr>
                <td style={{ opacity: 0.7 }}>Z-score</td>
                <td>
                  {z !== null ? (
                    <b
                      style={{
                        color:
                          Math.abs(z) >= 2
                            ? "#f66" // 이상치 강조
                            : "#fff",
                      }}
                    >
                      {z.toFixed(2)}
                    </b>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
              <tr>
                <td colSpan={2}>
                  <hr
                    style={{
                      border: "none",
                      borderTop: "1px solid #333",
                    }}
                  />
                </td>
              </tr>
            </>
          )}

          {/* ===== Window statistics ===== */}
          <tr><td>Min</td><td>{stats.min.toFixed(1)}</td></tr>
          <tr><td>Max</td><td>{stats.max.toFixed(1)}</td></tr>
          <tr><td>Mean</td><td>{stats.mean.toFixed(2)}</td></tr>
          <tr><td>Median</td><td>{stats.median.toFixed(1)}</td></tr>
          <tr><td>Std</td><td>{stats.std.toFixed(2)}</td></tr>
          <tr><td>N</td><td>{stats.count}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

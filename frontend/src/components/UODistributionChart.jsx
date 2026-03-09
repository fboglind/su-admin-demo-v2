import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { predsToChartData, UO_META } from "../utils/uo";

/**
 * Horizontal bar chart showing UO distribution predictions.
 *
 * Props:
 *   predictions: { 2434: value, ... }
 *   type: "binary" | "distributional" | "percentage"
 *   title: string
 *   height: number (default 280)
 *   showValues: boolean
 */
export default function UODistributionChart({
  predictions,
  type = "binary",
  title,
  height = 280,
  showValues = true,
}) {
  if (!predictions || Object.keys(predictions).length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
        No predictions available
      </div>
    );
  }

  const data = predsToChartData(predictions, type);
  const maxVal = Math.max(...data.map((d) => d.value), 1);

  // For binary: values are 0/1. For distributional: percentages 0-100
  const isPercentage = type === "distributional" || type === "percentage";
  const domainMax = isPercentage ? Math.min(100, Math.ceil(maxVal / 10) * 10 + 10) : 1.1;

  const formatValue = (val) => {
    if (isPercentage) return `${val.toFixed(1)}%`;
    if (type === "binary") return val >= 0.5 ? "✓" : "—";
    return val.toFixed(2);
  };

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 shadow-lg text-sm">
        <p className="font-medium" style={{ color: d.color }}>
          {d.abbr} — {d.name}
        </p>
        <p className="text-gray-600">{formatValue(d.value)}</p>
      </div>
    );
  };

  return (
    <div>
      {title && (
        <h3 className="text-sm font-semibold text-gray-700 mb-2">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: showValues ? 48 : 8 }}>
          <XAxis
            type="number"
            domain={[0, domainMax]}
            tickFormatter={(v) => (isPercentage ? `${v}%` : v)}
            tick={{ fontSize: 11, fill: "#9CA3AF" }}
          />
          <YAxis
            type="category"
            dataKey="abbr"
            width={32}
            tick={{ fontSize: 12, fontWeight: 500 }}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#F3F4F6" }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
            {data.map((entry) => (
              <Cell key={entry.code} fill={entry.color} fillOpacity={entry.value > 0 ? 0.85 : 0.15} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

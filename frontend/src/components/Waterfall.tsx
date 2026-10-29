import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Deduction } from "../types";

const COLOR: Record<string, string> = { critical: "#b3261e", high: "#c4521a", medium: "#b36b00", low: "#5b6b7a", info: "#6a7fa0" };

/** Every point deducted, one bar per named finding. They add up to 100 minus the score. */
export function Waterfall({ deductions }: { deductions: Deduction[] }) {
  const rows = deductions.slice(0, 12).map((d) => ({ name: `${d.code.replace(/_/g, " ")}${d.columns.length ? ` (${d.columns.slice(0, 2).join(", ")})` : ""}`, points: d.points, sev: d.severity }));
  const rest = deductions.slice(12).reduce((s, d) => s + d.points, 0);
  if (rest > 0) rows.push({ name: `${deductions.length - 12} smaller findings`, points: Math.round(rest * 100) / 100, sev: "low" });
  if (!rows.length) return <p className="muted">No points were deducted.</p>;
  return (
    <div role="img" aria-label={`Deductions: ${rows.map((r) => `${r.name} minus ${r.points}`).join("; ")}`} style={{ height: 28 * rows.length + 30 }}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24 }}>
          <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={230} tick={{ fill: "var(--fg)", fontSize: 12 }} />
          <Tooltip formatter={(v: number) => [`-${v} points`, "Deducted"]} />
          <Bar dataKey="points" radius={[0, 6, 6, 0]}>{rows.map((r, i) => <Cell key={i} fill={COLOR[r.sev]} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

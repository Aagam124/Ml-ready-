import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

/** vif: { [feature]: number | null }  (null means the backend measured infinite VIF - perfect collinearity) */
export default function VifChart({ vif, threshold }) {
  const data = Object.entries(vif)
    .map(([feature, value]) => ({ feature, value: value === null ? null : value }))
    .sort((a, b) => (b.value ?? Infinity) - (a.value ?? Infinity));

  if (data.length === 0) return null;

  // Cap the axis at a readable value; infinite/very-high VIF still renders as
  // a full-width bar with an exact label instead of stretching the chart.
  const finiteValues = data.map((d) => d.value).filter((v) => v !== null);
  const axisMax = Math.max(threshold * 2, ...(finiteValues.length ? finiteValues : [threshold * 2]));

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 32 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" horizontal={false} />
        <XAxis type="number" domain={[0, axisMax]} tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} stroke="var(--ink-muted)" />
        <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 12 }} stroke="var(--ink-muted)" />
        <ReferenceLine x={threshold} stroke="var(--ink-muted)" strokeDasharray="4 4" />
        <Tooltip
          formatter={(value, _name, entry) => [entry.payload.value === null ? '∞ (perfectly collinear)' : value, 'VIF']}
          contentStyle={{ fontSize: 12, borderRadius: 3, borderColor: 'var(--line)' }}
        />
        <Bar dataKey={(d) => (d.value === null ? axisMax : d.value)} radius={[0, 2, 2, 0]} barSize={16}>
          {data.map((entry) => (
            <Cell key={entry.feature} fill={(entry.value ?? Infinity) >= threshold ? 'var(--critical)' : 'var(--accent)'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

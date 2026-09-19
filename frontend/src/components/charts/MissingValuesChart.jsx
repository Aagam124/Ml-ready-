import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

export default function MissingValuesChart({ columnsWithMissing }) {
  const data = Object.entries(columnsWithMissing)
    .map(([column, info]) => ({ column, percentage: info.percentage }))
    .sort((a, b) => b.percentage - a.percentage);

  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" horizontal={false} />
        <XAxis type="number" unit="%" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} stroke="var(--ink-muted)" />
        <YAxis
          type="category" dataKey="column" width={120}
          tick={{ fontSize: 12 }} stroke="var(--ink-muted)"
        />
        <Tooltip
          formatter={(value) => [`${value}%`, 'Missing']}
          contentStyle={{ fontSize: 12, borderRadius: 3, borderColor: 'var(--line)' }}
        />
        <Bar dataKey="percentage" fill="var(--warn)" radius={[0, 2, 2, 0]} barSize={16} />
      </BarChart>
    </ResponsiveContainer>
  );
}

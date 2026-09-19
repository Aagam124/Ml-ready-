import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer } from 'recharts';

/** before/after: { [metric]: number } */
export default function BeforeAfterChart({ before, after }) {
  const metrics = Object.keys(before).filter((k) => typeof before[k] === 'number');
  const data = metrics.map((metric) => ({
    metric,
    Before: before[metric],
    After: after ? after[metric] : undefined,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ left: 0, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" vertical={false} />
        <XAxis dataKey="metric" tick={{ fontSize: 12 }} stroke="var(--ink-muted)" />
        <YAxis tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} stroke="var(--ink-muted)" />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 3, borderColor: 'var(--line)' }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Before" fill="var(--ink-muted)" radius={[2, 2, 0, 0]} barSize={28} />
        {after && <Bar dataKey="After" fill="var(--accent)" radius={[2, 2, 0, 0]} barSize={28} />}
      </BarChart>
    </ResponsiveContainer>
  );
}

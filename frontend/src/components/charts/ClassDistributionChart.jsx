import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';

export default function ClassDistributionChart({ classDistribution }) {
  const entries = Object.entries(classDistribution);
  const max = Math.max(...entries.map(([, count]) => count));
  const data = entries
    .map(([className, count]) => ({ className, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 40)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} stroke="var(--ink-muted)" />
        <YAxis type="category" dataKey="className" width={100} tick={{ fontSize: 12 }} stroke="var(--ink-muted)" />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 3, borderColor: 'var(--line)' }} />
        <Bar dataKey="count" radius={[0, 2, 2, 0]} barSize={18}>
          {data.map((entry) => (
            <Cell key={entry.className} fill={entry.count === max ? 'var(--accent)' : 'var(--warn)'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

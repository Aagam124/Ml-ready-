import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function DataTypesChart({ numericalCount, categoricalCount }) {
  const data = [
    { name: 'Numerical', value: numericalCount },
    { name: 'Categorical', value: categoricalCount },
  ].filter((d) => d.value > 0);

  const COLORS = ['var(--accent)', 'var(--warn)'];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={48} outerRadius={72} paddingAngle={2}>
          {data.map((entry, i) => (
            <Cell key={entry.name} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 3, borderColor: 'var(--line)' }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

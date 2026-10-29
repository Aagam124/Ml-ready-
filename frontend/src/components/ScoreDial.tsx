const COLORS = { Excellent: "var(--good)", Good: "var(--ok)", Fair: "var(--warn)", Poor: "var(--bad)" } as const;

export function ScoreDial({ value, band, provisional }: { value: number; band: keyof typeof COLORS; provisional?: boolean }) {
  const r = 78, c = 2 * Math.PI * r;
  return (
    <figure className="dial" aria-label={`Readiness score ${Math.round(value)} out of 100, ${band}${provisional ? ", provisional" : ""}`}>
      <svg viewBox="0 0 200 200" width="200" height="200" role="img" aria-hidden="true">
        <circle cx="100" cy="100" r={r} fill="none" stroke="var(--line)" strokeWidth="14" />
        <circle cx="100" cy="100" r={r} fill="none" stroke={COLORS[band]} strokeWidth="14" strokeLinecap="round" strokeDasharray={c}
          strokeDashoffset={c * (1 - Math.max(0, Math.min(100, value)) / 100)} transform="rotate(-90 100 100)" className="dial-arc" />
        <text x="100" y="108" textAnchor="middle" fontSize="54" fontWeight="700" fill="var(--fg)">{Math.round(value)}</text>
        <text x="100" y="136" textAnchor="middle" fontSize="16" fill="var(--muted)">{band}{provisional ? " (provisional)" : ""}</text>
      </svg>
    </figure>
  );
}

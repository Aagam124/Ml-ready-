/** status: 'good' | 'warn' | 'critical' | 'neutral' */
export default function StatusBadge({ status = 'neutral', children }) {
  return (
    <span className={`status ${status}`}>
      <span className="status-dot" />
      {children}
    </span>
  );
}

import { useEffect, useState } from 'react';
import { Link, useOutletContext } from 'react-router-dom';
import { apiErrorMessage, cleanDataset, getReport } from '../lib/api';
import EmptyState from '../components/EmptyState';

const OP_TYPES = [
  { value: 'impute_missing', label: 'Impute missing values' },
  { value: 'drop_rows_with_missing', label: 'Drop rows with missing values' },
  { value: 'drop_columns', label: 'Drop column(s)' },
  { value: 'remove_duplicates', label: 'Remove duplicate rows' },
  { value: 'handle_outliers', label: 'Handle outliers' },
];

function blankOp() {
  return { key: crypto.randomUUID(), type: 'impute_missing', column: '', strategy: 'mean', method: 'cap', columns: [] };
}

export default function Cleaning() {
  const { dataset, refetchDataset } = useOutletContext();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [ops, setOps] = useState([blankOp()]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    getReport(dataset.id).then((r) => setReport(r.report)).catch(() => setReport(null)).finally(() => setLoading(false));
  }, [dataset.id]);

  if (loading) return null;
  if (!report) {
    return (
      <EmptyState
        title="No analysis yet"
        description="Run analysis from the Overview tab first, so you know what needs cleaning."
        action={<Link to={`/datasets/${dataset.id}`}><button>Go to Overview</button></Link>}
      />
    );
  }

  const columns = report.profile.columns.map((c) => c.column);
  const numericColumns = report.profile.numerical_columns;

  function updateOp(key, patch) {
    setOps((prev) => prev.map((op) => (op.key === key ? { ...op, ...patch } : op)));
  }

  function addOp() {
    setOps((prev) => [...prev, blankOp()]);
  }

  function removeOp(key) {
    setOps((prev) => prev.filter((op) => op.key !== key));
  }

  function toPayload(op) {
    switch (op.type) {
      case 'impute_missing': return { type: op.type, column: op.column, strategy: op.strategy };
      case 'drop_rows_with_missing': return { type: op.type, columns: op.column ? [op.column] : undefined };
      case 'drop_columns': return { type: op.type, columns: op.columns };
      case 'remove_duplicates': return { type: op.type };
      case 'handle_outliers': return { type: op.type, column: op.column, method: op.method };
      default: return { type: op.type };
    }
  }

  async function handleApply() {
    setSubmitting(true);
    setError('');
    setResult(null);
    try {
      const payload = ops.map(toPayload);
      const entry = await cleanDataset(dataset.id, payload);
      setResult(entry);
      refetchDataset();
    } catch (err) {
      setError(apiErrorMessage(err, 'Cleaning failed.'));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      {report.recommendations.length > 0 && (
        <div className="section">
          <div className="section-title"><h2>Recommendations from the last analysis</h2></div>
          <div className="panel" style={{ padding: '4px 20px' }}>
            <ul className="recommendation-list">
              {report.recommendations.slice(0, 4).map((rec, i) => (
                <li key={i} className="recommendation-item">
                  <span className="recommendation-index">{String(i + 1).padStart(2, '0')}</span>{rec}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-title"><h2>Cleaning operations</h2></div>
        <div className="panel" style={{ padding: '4px 20px' }}>
          {ops.map((op) => (
            <div key={op.key} className="op-row">
              <div className="field">
                <label>Operation</label>
                <select value={op.type} onChange={(e) => updateOp(op.key, { type: e.target.value })}>
                  {OP_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
                </select>
              </div>

              {op.type === 'impute_missing' && (
                <>
                  <div className="field">
                    <label>Column</label>
                    <select value={op.column} onChange={(e) => updateOp(op.key, { column: e.target.value })}>
                      <option value="">Select…</option>
                      {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="field">
                    <label>Strategy</label>
                    <select value={op.strategy} onChange={(e) => updateOp(op.key, { strategy: e.target.value })}>
                      <option value="mean">Mean</option>
                      <option value="median">Median</option>
                      <option value="mode">Mode</option>
                    </select>
                  </div>
                </>
              )}

              {op.type === 'drop_rows_with_missing' && (
                <div className="field">
                  <label>Column (optional — blank means any column)</label>
                  <select value={op.column} onChange={(e) => updateOp(op.key, { column: e.target.value })}>
                    <option value="">Any column</option>
                    {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )}

              {op.type === 'drop_columns' && (
                <div className="field">
                  <label>Column(s) to drop</label>
                  <select
                    multiple value={op.columns}
                    onChange={(e) => updateOp(op.key, { columns: Array.from(e.target.selectedOptions, (o) => o.value) })}
                    style={{ height: 72 }}
                  >
                    {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )}

              {op.type === 'handle_outliers' && (
                <>
                  <div className="field">
                    <label>Column</label>
                    <select value={op.column} onChange={(e) => updateOp(op.key, { column: e.target.value })}>
                      <option value="">Select…</option>
                      {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="field">
                    <label>Method</label>
                    <select value={op.method} onChange={(e) => updateOp(op.key, { method: e.target.value })}>
                      <option value="cap">Cap (winsorize) to IQR bounds</option>
                      <option value="remove">Remove flagged rows</option>
                      <option value="keep">Keep as-is</option>
                    </select>
                  </div>
                </>
              )}

              <button className="secondary op-remove" onClick={() => removeOp(op.key)} disabled={ops.length === 1}>
                Remove
              </button>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12, display: 'flex', gap: 10 }}>
          <button className="secondary" onClick={addOp}>+ Add operation</button>
          <button onClick={handleApply} disabled={submitting}>{submitting ? 'Applying…' : 'Apply cleaning'}</button>
        </div>
        {error && <p className="error-text">{error}</p>}
      </div>

      {result && (
        <div className="section">
          <div className="section-title"><h2>Result</h2></div>
          <div className="panel" style={{ padding: 16 }}>
            <p>
              Resulting shape: <strong className="mono">{result.resulting_shape.n_rows}</strong> rows ×{' '}
              <strong className="mono">{result.resulting_shape.n_columns}</strong> columns
            </p>
            <ul className="deduction-list" style={{ marginTop: 10 }}>
              {result.operations.map((op, i) => (
                <li key={i} className="deduction-row">
                  <span>{op.type}{op.column ? ` — ${op.column}` : ''}</span>
                  <span className="mono text-muted">
                    {op.rows_affected !== undefined ? `${op.rows_affected} rows` : ''}
                    {op.note ? op.note : ''}
                  </span>
                </li>
              ))}
            </ul>
            <p className="section-note">
              The original file was left untouched — a separate cleaned copy was saved. Head to Model Comparison to
              train a before/after model and see whether this actually helped.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

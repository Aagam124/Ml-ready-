import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { downloadReport, getFullReport } from '../lib/api';
import { formatDate, scoreCategoryToStatus } from '../lib/format';
import StatusBadge from '../components/StatusBadge';

export default function FullReport() {
  const { dataset } = useOutletContext();
  const [bundle, setBundle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    getFullReport(dataset.id).then(setBundle).finally(() => setLoading(false));
  }, [dataset.id]);

  async function handleExport(format) {
    setExporting(format);
    setError('');
    try {
      await downloadReport(dataset.id, format);
    } catch {
      setError(`Could not export as ${format.toUpperCase()}.`);
    } finally {
      setExporting('');
    }
  }

  if (loading || !bundle) return null;

  const { analysis, cleaning_history: cleaningHistory, model_comparison: modelComparison } = bundle;

  return (
    <div>
      <div className="section-title" style={{ marginBottom: 20 }}>
        <h2>Full report</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="secondary" onClick={() => handleExport('html')} disabled={!!exporting}>
            {exporting === 'html' ? 'Exporting…' : 'Export HTML'}
          </button>
          <button className="secondary" onClick={() => handleExport('pdf')} disabled={!!exporting}>
            {exporting === 'pdf' ? 'Exporting…' : 'Export PDF'}
          </button>
        </div>
      </div>
      {error && <p className="error-text">{error}</p>}

      <div className="section">
        <h3 style={{ marginBottom: 8 }}>Dataset</h3>
        <div className="panel" style={{ padding: 16 }}>
          <p>{dataset.filename} — {formatDate(dataset.uploaded_at)}</p>
        </div>
      </div>

      <div className="section">
        <h3 style={{ marginBottom: 8 }}>Readiness &amp; recommendations</h3>
        <div className="panel" style={{ padding: 16 }}>
          {analysis ? (
            <>
              <StatusBadge status={scoreCategoryToStatus(analysis.readiness_score.category)}>
                {analysis.readiness_score.score}/100 — {analysis.readiness_score.category}
              </StatusBadge>
              <ul className="recommendation-list" style={{ marginTop: 12 }}>
                {analysis.recommendations.slice(0, 5).map((rec, i) => (
                  <li key={i} className="recommendation-item">
                    <span className="recommendation-index">{String(i + 1).padStart(2, '0')}</span>{rec}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-muted">Not analyzed yet.</p>
          )}
        </div>
      </div>

      <div className="section">
        <h3 style={{ marginBottom: 8 }}>Cleaning history</h3>
        <div className="panel" style={{ padding: 16 }}>
          {cleaningHistory.length === 0 ? (
            <p className="text-muted">No cleaning applied yet.</p>
          ) : (
            cleaningHistory.map((entry, i) => (
              <p key={entry.id} className="mono" style={{ fontSize: 12.5, marginBottom: 6 }}>
                Run {i + 1}: {entry.operations.map((op) => op.type).join(', ')} → {entry.resulting_shape.n_rows} rows
              </p>
            ))
          )}
        </div>
      </div>

      <div className="section">
        <h3 style={{ marginBottom: 8 }}>Model comparison</h3>
        <div className="panel" style={{ padding: 16 }}>
          {modelComparison ? (
            <p>
              {modelComparison.before.model_type}: accuracy/primary metric before vs after — see chart on the Model
              Comparison tab for the full breakdown.
            </p>
          ) : (
            <p className="text-muted">No model trained yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

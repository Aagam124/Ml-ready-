import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { listCleaningHistory, listModelResultsHistory, listReports } from '../lib/api';
import { formatDate } from '../lib/format';
import EmptyState from '../components/EmptyState';

export default function AnalysisHistory() {
  const { dataset } = useOutletContext();
  const [reports, setReports] = useState([]);
  const [cleaning, setCleaning] = useState([]);
  const [modelRuns, setModelRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([listReports(dataset.id), listCleaningHistory(dataset.id), listModelResultsHistory(dataset.id)])
      .then(([r, c, m]) => { setReports(r); setCleaning(c); setModelRuns(m); })
      .finally(() => setLoading(false));
  }, [dataset.id]);

  if (loading) return null;

  if (reports.length === 0 && cleaning.length === 0 && modelRuns.length === 0) {
    return <EmptyState title="No history yet" description="Run an analysis, cleaning operation, or training run to see it recorded here." />;
  }

  return (
    <div>
      <div className="section">
        <div className="section-title"><h2>Analysis runs</h2><span className="badge-count">{reports.length}</span></div>
        {reports.length === 0 ? (
          <p className="text-muted">None yet.</p>
        ) : (
          <div className="panel">
            {[...reports].reverse().map((r) => (
              <div key={r.id} className="dataset-row">
                <div className="dataset-row-main">
                  <div className="dataset-row-name">Score: {r.report.readiness_score.score}/100 — {r.report.readiness_score.category}</div>
                  <div className="dataset-row-meta">{formatDate(r.created_at)}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title"><h2>Cleaning operations</h2><span className="badge-count">{cleaning.length}</span></div>
        {cleaning.length === 0 ? (
          <p className="text-muted">None yet.</p>
        ) : (
          <div className="panel">
            {[...cleaning].reverse().map((entry) => (
              <div key={entry.id} className="dataset-row" style={{ alignItems: 'flex-start' }}>
                <div className="dataset-row-main">
                  <div className="dataset-row-name">
                    {entry.operations.map((op) => op.type).join(', ')}
                  </div>
                  <div className="dataset-row-meta">
                    {formatDate(entry.created_at)} → {entry.resulting_shape.n_rows} rows × {entry.resulting_shape.n_columns} cols
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title"><h2>Model training runs</h2><span className="badge-count">{modelRuns.length}</span></div>
        {modelRuns.length === 0 ? (
          <p className="text-muted">None yet.</p>
        ) : (
          <div className="panel">
            {[...modelRuns].reverse().map((run) => (
              <div key={run.id} className="dataset-row">
                <div className="dataset-row-main">
                  <div className="dataset-row-name">
                    {run.stage === 'before' ? 'Before cleaning' : 'After cleaning'} — {run.model_type}
                    {run.details?.target_column ? ` on "${run.details.target_column}"` : ''}
                  </div>
                  <div className="dataset-row-meta">
                    {formatDate(run.created_at)} · {Object.entries(run.metrics).map(([k, v]) => `${k}: ${v}`).join(', ')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

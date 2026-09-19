import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { analyzeDataset, apiErrorMessage, getReport } from '../lib/api';
import { formatNumber } from '../lib/format';
import ScoreDial from '../components/ScoreDial';
import DataTypesChart from '../components/charts/DataTypesChart';
import EmptyState from '../components/EmptyState';

export default function DatasetOverview() {
  const { dataset, refetchDataset } = useOutletContext();
  const [report, setReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(true);
  const [targetColumn, setTargetColumn] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getReport(dataset.id)
      .then((r) => setReport(r.report))
      .catch(() => setReport(null))
      .finally(() => setLoadingReport(false));
  }, [dataset.id]);

  async function handleAnalyze() {
    setAnalyzing(true);
    setError('');
    try {
      const result = await analyzeDataset(dataset.id, targetColumn || undefined);
      setReport(result.report);
      refetchDataset();
    } catch (err) {
      setError(apiErrorMessage(err, 'Analysis failed.'));
    } finally {
      setAnalyzing(false);
    }
  }

  if (loadingReport) return null;

  const columns = dataset.metadata?.column_names || [];
  const semanticByColumn = Object.fromEntries(
    (report?.profile?.column_details || []).map((d) => [d.column, d])
  );

  return (
    <div>
      <div className="section">
        <div className="section-title"><h2>Run analysis</h2></div>
        <div className="panel" style={{ padding: 20 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div className="field" style={{ marginBottom: 0, flex: '1 1 220px' }}>
              <label htmlFor="target">Target column (optional — enables class imbalance &amp; leakage checks)</label>
              <select id="target" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                <option value="">None selected</option>
                {columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <button onClick={handleAnalyze} disabled={analyzing}>
              {analyzing ? 'Analyzing…' : report ? 'Re-run analysis' : 'Analyze dataset'}
            </button>
          </div>
          {error && <p className="error-text">{error}</p>}
        </div>
      </div>

      {!report && (
        <EmptyState
          title="Not analyzed yet"
          description="Run analysis to get a readiness score, data quality report, and ML risk analysis for this dataset."
        />
      )}

      {report && (
        <>
          <div className="section">
            <div className="section-title"><h2>Readiness score</h2></div>
            <div className="panel" style={{ padding: 24 }}>
              <ScoreDial score={report.readiness_score.score} category={report.readiness_score.category} />
              {report.readiness_score.deductions?.length > 0 && (
                <ul className="deduction-list" style={{ marginTop: 20, borderTop: '1px solid var(--line)', paddingTop: 16 }}>
                  {report.readiness_score.deductions.map((d, i) => (
                    <li key={i} className="deduction-row">
                      <span>{d.issue}</span>
                      <span className="deduction-points">{d.points}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          <div className="section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div>
              <div className="section-title"><h2>Column types</h2></div>
              <div className="panel" style={{ padding: 12 }}>
                <DataTypesChart
                  numericalCount={report.profile.numerical_columns.length}
                  categoricalCount={report.profile.categorical_columns.length}
                />
              </div>
            </div>
            <div>
              <div className="section-title"><h2>Dataset shape</h2></div>
              <div className="data-strip" style={{ marginBottom: 0, flexDirection: 'column' }}>
                <div className="data-strip-item" style={{ borderRight: 'none', borderBottom: '1px solid var(--line)' }}>
                  <div className="data-strip-label">Duplicate rows</div>
                  <div className="data-strip-value small">{report.profile.duplicate_rows.count} ({report.profile.duplicate_rows.percentage}%)</div>
                </div>
                <div className="data-strip-item" style={{ borderRight: 'none' }}>
                  <div className="data-strip-label">Numerical / categorical columns</div>
                  <div className="data-strip-value small">
                    {report.profile.numerical_columns.length} / {report.profile.categorical_columns.length}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="section">
            <div className="section-title"><h2>Column profile</h2></div>
            <div className="panel" style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column</th><th>Type</th><th className="num">Missing %</th>
                    <th className="num">Unique</th><th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {report.profile.columns.map((col) => {
                    const detail = semanticByColumn[col.column];
                    const typeLabel = detail
                      ? detail.semantic_type.charAt(0).toUpperCase() + detail.semantic_type.slice(1)
                      : col.kind;
                    return (
                      <tr key={col.column}>
                        <td>{col.column}</td>
                        <td title={detail ? detail.semantic_evidence : undefined}>{typeLabel}</td>
                        <td className="num">{col.missing_pct}%</td>
                        <td className="num">{formatNumber(col.unique_count)}</td>
                        <td className="mono" style={{ fontSize: 12 }}>
                          {col.kind === 'numerical'
                            ? `min ${col.min} · mean ${col.mean?.toFixed?.(2) ?? col.mean} · max ${col.max}`
                            : Object.keys(col.top_values || {}).slice(0, 3).join(', ')}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

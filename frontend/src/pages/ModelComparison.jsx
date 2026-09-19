import { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { getModelResults, trainModel, apiErrorMessage } from '../lib/api';
import BeforeAfterChart from '../components/charts/BeforeAfterChart';

export default function ModelComparison() {
  const { dataset } = useOutletContext();
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [targetColumn, setTargetColumn] = useState('');
  const [error, setError] = useState('');

  const columns = dataset.metadata?.column_names || [];

  useEffect(() => {
    getModelResults(dataset.id)
      .then(setComparison)
      .catch(() => {}) // 404 "no model trained yet" - comparison stays null, empty state below handles it
      .finally(() => setLoading(false));
  }, [dataset.id]);

  async function handleTrain() {
    if (!targetColumn) {
      setError('Choose a target column first.');
      return;
    }
    setTraining(true);
    setError('');
    try {
      const result = await trainModel(dataset.id, targetColumn);
      setComparison(result);
    } catch (err) {
      setError(apiErrorMessage(err, 'Training failed.'));
    } finally {
      setTraining(false);
    }
  }

  if (loading) return null;

  return (
    <div>
      <div className="section">
        <div className="section-title"><h2>Before vs. after cleaning</h2></div>
        <div className="panel" style={{ padding: 20 }}>
          {!comparison?.before ? (
            <>
              <p className="text-muted" style={{ marginBottom: 12 }}>
                Training runs a baseline classifier/regressor on the original data, then again on the cleaned
                version (if you've run cleaning), so you can see whether cleaning actually helped.
              </p>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                <div className="field" style={{ marginBottom: 0, flex: '1 1 220px' }}>
                  <label htmlFor="target">Target column</label>
                  <select id="target" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                    <option value="">Select…</option>
                    {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <button onClick={handleTrain} disabled={training}>
                  {training ? 'Training…' : 'Train baseline model'}
                </button>
              </div>
              {error && <p className="error-text">{error}</p>}
            </>
          ) : (
            <>
              <p className="section-note" style={{ marginBottom: 12 }}>
                Trained on target: <strong className="mono">{comparison.target_column}</strong>
              </p>
              <BeforeAfterChart before={comparison.before.metrics} after={comparison.after?.metrics} />
              {comparison.deltas && Object.keys(comparison.deltas).length > 0 && (
                <ul className="deduction-list" style={{ marginTop: 16 }}>
                  {Object.entries(comparison.deltas).map(([metric, delta]) => (
                    <li key={metric} className="deduction-row">
                      <span>{metric}</span>
                      <span className="mono" style={{ color: delta >= 0 ? 'var(--good)' : 'var(--critical)' }}>
                        {delta >= 0 ? '+' : ''}{delta}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              {!comparison.after && comparison.after_skipped_reason && (
                <p className="section-note">{comparison.after_skipped_reason}</p>
              )}
              {comparison.after && (
                <p className="section-note">
                  Cleaning doesn't always improve performance — if a metric moved the wrong way, that's shown honestly above.
                </p>
              )}
              <div style={{ marginTop: 16, display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                <div className="field" style={{ marginBottom: 0, flex: '1 1 220px' }}>
                  <label htmlFor="target-retrain">Re-train with a different target</label>
                  <select id="target-retrain" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                    <option value="">Select…</option>
                    {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <button className="secondary" onClick={handleTrain} disabled={training}>
                  {training ? 'Training…' : 'Train again'}
                </button>
              </div>
              {error && <p className="error-text">{error}</p>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

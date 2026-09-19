import { useEffect, useState } from 'react';
import { Link, useOutletContext } from 'react-router-dom';
import { getReport } from '../lib/api';
import MissingValuesChart from '../components/charts/MissingValuesChart';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';

function statusFor(count, warnAt = 1, criticalAt = 5) {
  if (count === 0) return 'good';
  if (count < criticalAt) return count < warnAt + 1 ? 'warn' : 'warn';
  return 'critical';
}

export default function DataQualityReport() {
  const { dataset } = useOutletContext();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getReport(dataset.id).then((r) => setReport(r.report)).catch(() => setReport(null)).finally(() => setLoading(false));
  }, [dataset.id]);

  if (loading) return null;
  if (!report) {
    return (
      <EmptyState
        title="No analysis yet"
        description="Run analysis from the Overview tab first."
        action={<Link to={`/datasets/${dataset.id}`}><button>Go to Overview</button></Link>}
      />
    );
  }

  const { quality, outliers } = report;
  const missingCols = Object.keys(quality.missing_values.columns_with_missing);
  const dtypeIssues = Object.entries(quality.dtype_issues);
  const nearConstant = Object.entries(quality.near_constant_columns);
  const highCard = Object.entries(quality.high_cardinality_columns);
  const sentinelCols = Object.entries(quality.invalid_values?.possible_missing_sentinels || {});
  const consensusCols = Object.entries(outliers.consensus || {});

  return (
    <div>
      <div className="section">
        <div className="section-title"><h2>Missing values</h2>
          <StatusBadge status={statusFor(missingCols.length)}>
            {quality.missing_values.overall_percentage}% of all cells
          </StatusBadge>
        </div>
        <div className="panel" style={{ padding: missingCols.length ? '16px 20px' : 24 }}>
          {missingCols.length ? (
            <MissingValuesChart columnsWithMissing={quality.missing_values.columns_with_missing} />
          ) : (
            <p className="text-muted">No missing values detected.</p>
          )}
        </div>
      </div>

      {sentinelCols.length > 0 && (
        <div className="section">
          <div className="section-title"><h2>Possible missing-value placeholders</h2>
            <StatusBadge status="warn">{sentinelCols.length} column(s)</StatusBadge>
          </div>
          <div className="panel" style={{ padding: 16 }}>
            <p className="section-note" style={{ marginBottom: 10 }}>
              These columns contain text values that often stand in for missing data (like &quot;N/A&quot; or
              &quot;unknown&quot;) but aren&apos;t empty cells, so they weren&apos;t counted above.
            </p>
            <ul className="deduction-list">
              {sentinelCols.map(([c, info]) => (
                <li key={c} className="deduction-row">
                  <span>{c}</span>
                  <span className="mono text-muted">{info.count} value(s): {info.values.join(', ')}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-title"><h2>Duplicate rows</h2>
          <StatusBadge status={statusFor(quality.duplicates.count, 1, 20)}>
            {quality.duplicates.count} rows ({quality.duplicates.percentage}%)
          </StatusBadge>
        </div>
      </div>

      <div className="section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div>
          <div className="section-title"><h2>Constant &amp; near-constant columns</h2></div>
          <div className="panel" style={{ padding: 16 }}>
            {quality.constant_columns.length === 0 && nearConstant.length === 0 ? (
              <p className="text-muted">None found.</p>
            ) : (
              <ul className="deduction-list">
                {quality.constant_columns.map((c) => (
                  <li key={c} className="deduction-row"><span>{c}</span><span className="text-muted">constant</span></li>
                ))}
                {nearConstant.map(([c, info]) => (
                  <li key={c} className="deduction-row">
                    <span>{c}</span>
                    <span className="mono text-muted">{(info.dominant_value_ratio * 100).toFixed(1)}% one value</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
        <div>
          <div className="section-title"><h2>Likely mistyped columns</h2></div>
          <div className="panel" style={{ padding: 16 }}>
            {dtypeIssues.length === 0 ? (
              <p className="text-muted">None found.</p>
            ) : (
              <ul className="deduction-list">
                {dtypeIssues.map(([c, info]) => (
                  <li key={c} className="deduction-row">
                    <span>{c}</span>
                    <span className="mono text-muted">{(info.convertible_ratio * 100).toFixed(0)}% look numeric</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-title"><h2>High-cardinality columns</h2></div>
        <div className="panel" style={{ padding: 16 }}>
          {highCard.length === 0 ? (
            <p className="text-muted">None found.</p>
          ) : (
            <ul className="deduction-list">
              {highCard.map(([c, info]) => (
                <li key={c} className="deduction-row">
                  <span>{c}</span>
                  <span className="mono text-muted">{info.unique_count} unique ({(info.unique_ratio * 100).toFixed(0)}%)</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title"><h2>Outliers</h2></div>
        <div className="panel" style={{ padding: 16 }}>
          {consensusCols.length > 0 ? (
            <>
              <p style={{ marginBottom: 10 }}>Flagged by more than one detection method (the strongest evidence):</p>
              <ul className="deduction-list" style={{ marginBottom: 14 }}>
                {consensusCols.map(([c, info]) => (
                  <li key={c} className="deduction-row">
                    <span>{c}</span>
                    <span className="mono text-muted">{info.methods.join(' + ')} ({Math.round(info.agreement * 100)}% agreement)</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-muted" style={{ marginBottom: 10 }}>No column was flagged by more than one method.</p>
          )}
          {outliers.isolation_forest.applied ? (
            <p>
              Isolation Forest flagged <strong>{outliers.isolation_forest.outlier_count}</strong> rows
              ({outliers.isolation_forest.percentage}%) as multivariate outliers across{' '}
              {outliers.isolation_forest.columns_used.length} numeric columns.
            </p>
          ) : (
            <p className="text-muted">{outliers.isolation_forest.reason}</p>
          )}
          <p className="section-note">{outliers.note}</p>
        </div>
      </div>
    </div>
  );
}

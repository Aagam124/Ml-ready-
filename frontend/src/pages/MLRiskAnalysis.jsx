import { Fragment, useEffect, useState } from 'react';
import { Link, useOutletContext } from 'react-router-dom';
import { getReport } from '../lib/api';
import ClassDistributionChart from '../components/charts/ClassDistributionChart';
import VifChart from '../components/charts/VifChart';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';

function severityToStatus(severity) {
  const s = (severity || '').toUpperCase();
  if (s === 'CRITICAL') return 'critical';
  if (s === 'HIGH') return 'critical';
  if (s === 'MEDIUM') return 'warn';
  return 'good';
}

function recommendationToStatus(rec) {
  const r = (rec || '').toUpperCase();
  if (r === 'REMOVE') return 'critical';
  if (r === 'REVIEW') return 'critical';
  if (r === 'IMPUTE' || r === 'TRANSFORM') return 'warn';
  return 'good';
}

export default function MLRiskAnalysis() {
  const { dataset } = useOutletContext();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedFeature, setExpandedFeature] = useState(null);

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

  const {
    class_imbalance,
    multicollinearity,
    potential_leakage,
    high_cardinality_features,
    train_test_distribution_diff,
    feature_risk_analysis,
    duplicate_columns,
    dimensionality_risk,
  } = report.ml_risks || {};

  const leakageCols = potential_leakage?.potential_leakage_columns || [];
  const hcFeatures = Object.entries(high_cardinality_features || {});
  const shiftedCols = train_test_distribution_diff?.applicable
    ? Object.entries(train_test_distribution_diff.columns_with_significant_shift || {})
    : [];

  const fra = feature_risk_analysis || null;
  const featureRows = fra?.features
    ? Object.entries(fra.features).sort((a, b) => (b[1].risk_score || 0) - (a[1].risk_score || 0))
    : [];
  const summary = fra?.summary || {};

  return (
    <div>
      {fra && (
        <div className="section">
          <div className="section-title">
            <h2>Feature risk analysis</h2>
            <StatusBadge status={
              (summary.critical || 0) > 0 ? 'critical'
                : (summary.high || 0) > 0 ? 'critical'
                  : (summary.medium || 0) > 0 ? 'warn'
                    : 'good'
            }>
              {(summary.critical || 0) + (summary.high || 0)} high/critical · {(summary.medium || 0)} medium · {(summary.low || 0)} low
            </StatusBadge>
          </div>

          <div className="panel" style={{ padding: 16, marginBottom: 12 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 13.5 }}>
              <span><strong>{summary.critical || 0}</strong> critical</span>
              <span><strong>{summary.high || 0}</strong> high</span>
              <span><strong>{summary.medium || 0}</strong> medium</span>
              <span><strong>{summary.low || 0}</strong> low</span>
              <span className="text-muted">|</span>
              <span><strong>{summary.remove || 0}</strong> remove</span>
              <span><strong>{summary.review || 0}</strong> review</span>
              <span><strong>{summary.transform || 0}</strong> transform</span>
              <span><strong>{summary.impute || 0}</strong> impute</span>
              <span><strong>{summary.keep || 0}</strong> keep</span>
            </div>
            {fra.methodology?.important_note && (
              <p className="section-note" style={{ marginTop: 10, marginBottom: 0 }}>
                {fra.methodology.important_note}
              </p>
            )}
          </div>

          <div className="panel" style={{ padding: 0, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13.5 }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border, #e5e7eb)' }}>
                  <th style={{ padding: '10px 14px' }}>Feature</th>
                  <th style={{ padding: '10px 14px' }}>Score</th>
                  <th style={{ padding: '10px 14px' }}>Severity</th>
                  <th style={{ padding: '10px 14px' }}>Action</th>
                  <th style={{ padding: '10px 14px' }}>Missing %</th>
                </tr>
              </thead>
              <tbody>
                {featureRows.map(([name, info]) => (
                  <Fragment key={name}>
                    <tr
                      onClick={() => setExpandedFeature(expandedFeature === name ? null : name)}
                      style={{
                        borderBottom: '1px solid var(--border, #f0f0f0)',
                        cursor: 'pointer',
                        background: expandedFeature === name ? 'var(--surface-2, #f9fafb)' : undefined,
                      }}
                    >
                      <td style={{ padding: '10px 14px', fontWeight: 500 }}>{name}</td>
                      <td style={{ padding: '10px 14px' }} className="mono">{info.risk_score}</td>
                      <td style={{ padding: '10px 14px' }}>
                        <StatusBadge status={severityToStatus(info.severity)}>{info.severity}</StatusBadge>
                      </td>
                      <td style={{ padding: '10px 14px' }}>
                        <StatusBadge status={recommendationToStatus(info.recommendation)}>
                          {info.recommendation}
                        </StatusBadge>
                      </td>
                      <td style={{ padding: '10px 14px' }} className="mono text-muted">
                        {info.missing_percentage ?? 0}%
                      </td>
                    </tr>
                    {expandedFeature === name && (
                      <tr>
                        <td colSpan={5} style={{ padding: '12px 14px 16px', background: 'var(--surface-2, #f9fafb)' }}>
                          <div style={{ display: 'grid', gap: 8, maxWidth: 720 }}>
                            <div>
                              <strong style={{ fontSize: 12.5, textTransform: 'uppercase', letterSpacing: 0.4 }}>Evidence</strong>
                              <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                                {(info.evidence || []).map((e, i) => (
                                  <li key={i} className="text-muted" style={{ fontSize: 13 }}>{e}</li>
                                ))}
                              </ul>
                            </div>
                            <div>
                              <strong style={{ fontSize: 12.5, textTransform: 'uppercase', letterSpacing: 0.4 }}>Suggested actions</strong>
                              <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                                {(info.suggested_actions || []).map((a, i) => (
                                  <li key={i} style={{ fontSize: 13 }}>{a}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
            {featureRows.length === 0 && (
              <p className="text-muted" style={{ padding: 16 }}>No feature risk data available. Re-run analysis after updating the backend.</p>
            )}
          </div>
        </div>
      )}

      {(dimensionality_risk || duplicate_columns) && (
        <div className="section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {dimensionality_risk && (
            <div>
              <div className="section-title">
                <h2>Dimensionality risk</h2>
                <StatusBadge status={
                  dimensionality_risk.risk === 'HIGH' ? 'critical'
                    : dimensionality_risk.risk === 'MEDIUM' ? 'warn' : 'good'
                }>
                  {dimensionality_risk.risk}
                </StatusBadge>
              </div>
              <div className="panel" style={{ padding: 16 }}>
                <p style={{ margin: 0 }}>{dimensionality_risk.message}</p>
                <p className="text-muted" style={{ margin: '8px 0 0', fontSize: 13 }}>
                  {dimensionality_risk.n_features} features / {dimensionality_risk.n_rows} rows
                  (ratio {dimensionality_risk.feature_to_sample_ratio})
                </p>
              </div>
            </div>
          )}
          {duplicate_columns && (
            <div>
              <div className="section-title">
                <h2>Duplicate columns</h2>
                <StatusBadge status={
                  (duplicate_columns.exact_duplicates?.length || 0) + (duplicate_columns.near_duplicates?.length || 0) > 0
                    ? 'warn' : 'good'
                }>
                  {(duplicate_columns.exact_duplicates?.length || 0)} exact · {(duplicate_columns.near_duplicates?.length || 0)} near
                </StatusBadge>
              </div>
              <div className="panel" style={{ padding: 16 }}>
                {(duplicate_columns.exact_duplicates?.length || 0) === 0
                  && (duplicate_columns.near_duplicates?.length || 0) === 0 ? (
                  <p className="text-muted" style={{ margin: 0 }}>No exact or near-duplicate columns detected.</p>
                ) : (
                  <ul className="deduction-list">
                    {(duplicate_columns.exact_duplicates || []).map((p, i) => (
                      <li key={`e-${i}`} className="deduction-row">
                        <span>{p.feature_1} ≡ {p.feature_2}</span>
                        <span className="mono text-muted">exact</span>
                      </li>
                    ))}
                    {(duplicate_columns.near_duplicates || []).map((p, i) => (
                      <li key={`n-${i}`} className="deduction-row">
                        <span>{p.feature_1} ≈ {p.feature_2}</span>
                        <span className="mono text-muted">r={p.correlation}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="section">
        <div className="section-title"><h2>Class imbalance</h2>
          {class_imbalance?.applicable && (
            <StatusBadge status={class_imbalance.is_imbalanced ? 'critical' : 'good'}>
              {class_imbalance.majority_minority_ratio}:1 majority:minority
            </StatusBadge>
          )}
        </div>
        <div className="panel" style={{ padding: 16 }}>
          {class_imbalance?.applicable ? (
            <ClassDistributionChart classDistribution={class_imbalance.class_distribution} />
          ) : (
            <p className="text-muted">{class_imbalance?.reason || 'Not available.'}</p>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title"><h2>Multicollinearity</h2>
          {(() => {
            const pairCount = multicollinearity?.high_correlation_pairs?.length || 0;
            const vifValues = Object.values(multicollinearity?.vif || {});
            const threshold = multicollinearity?.vif_threshold_used;
            const highVifCount = vifValues.filter((v) => v === null || (threshold != null && v >= threshold)).length;
            const flagged = pairCount > 0 || highVifCount > 0;
            return (
              <StatusBadge status={flagged ? 'warn' : 'good'}>
                {pairCount} correlated pair(s), {highVifCount} high-VIF feature(s)
              </StatusBadge>
            );
          })()}
        </div>
        <div className="panel" style={{ padding: 16 }}>
          {Object.keys(multicollinearity?.vif || {}).length > 0 ? (
            <>
              <VifChart vif={multicollinearity.vif} threshold={multicollinearity.vif_threshold_used} />
              <p className="section-note">Dashed line marks the VIF threshold ({multicollinearity.vif_threshold_used}).</p>
            </>
          ) : (
            <p className="text-muted">{multicollinearity?.note || 'Not enough numeric columns to evaluate.'}</p>
          )}
          {multicollinearity?.high_correlation_pairs?.length > 0 && (
            <ul className="deduction-list" style={{ marginTop: 12 }}>
              {multicollinearity.high_correlation_pairs.map((pair) => (
                <li key={`${pair.feature_a}-${pair.feature_b}`} className="deduction-row">
                  <span>{pair.feature_a} ↔ {pair.feature_b}</span>
                  <span className="mono">{pair.correlation}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title"><h2>Potential data leakage</h2>
          <StatusBadge status={leakageCols.length ? 'critical' : 'good'}>
            {leakageCols.length} flagged column(s)
          </StatusBadge>
        </div>
        <div className="panel" style={{ padding: 16 }}>
          {leakageCols.length === 0 ? (
            <p className="text-muted">No suspicious columns found.</p>
          ) : (
            <ul className="deduction-list">
              {leakageCols.map((flag) => (
                <li key={flag.column} className="deduction-row" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 4 }}>
                  <strong>{flag.column}</strong>
                  <span className="text-muted" style={{ fontSize: 12.5 }}>{flag.reasons.join(' ')}</span>
                </li>
              ))}
            </ul>
          )}
          <p className="section-note">{potential_leakage?.disclaimer}</p>
        </div>
      </div>

      <div className="section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div>
          <div className="section-title"><h2>High-cardinality features</h2></div>
          <div className="panel" style={{ padding: 16 }}>
            {hcFeatures.length === 0 ? (
              <p className="text-muted">None found.</p>
            ) : (
              <ul className="deduction-list">
                {hcFeatures.map(([c, info]) => (
                  <li key={c} className="deduction-row">
                    <span>{c}</span>
                    <span className="mono text-muted">{info.unique_count} unique</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
        <div>
          <div className="section-title"><h2>Train/test distribution shift</h2></div>
          <div className="panel" style={{ padding: 16 }}>
            {!train_test_distribution_diff?.applicable ? (
              <p className="text-muted">{train_test_distribution_diff?.reason}</p>
            ) : shiftedCols.length === 0 ? (
              <p className="text-muted">No significant shift detected (split {train_test_distribution_diff.split_used}).</p>
            ) : (
              <ul className="deduction-list">
                {shiftedCols.map(([c, info]) => (
                  <li key={c} className="deduction-row">
                    <span>{c}</span>
                    <span className="mono text-muted">p={info.p_value}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
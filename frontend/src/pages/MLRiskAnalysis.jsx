import { useEffect, useState } from 'react';
import { Link, useOutletContext } from 'react-router-dom';
import { getReport } from '../lib/api';
import ClassDistributionChart from '../components/charts/ClassDistributionChart';
import VifChart from '../components/charts/VifChart';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';

export default function MLRiskAnalysis() {
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

  const { class_imbalance, multicollinearity, potential_leakage, high_cardinality_features, train_test_distribution_diff } = report.ml_risks;
  const leakageCols = potential_leakage?.potential_leakage_columns || [];
  const hcFeatures = Object.entries(high_cardinality_features || {});
  const shiftedCols = train_test_distribution_diff?.applicable
    ? Object.entries(train_test_distribution_diff.columns_with_significant_shift || {})
    : [];

  return (
    <div>
      <div className="section">
        <div className="section-title"><h2>Class imbalance</h2>
          {class_imbalance.applicable && (
            <StatusBadge status={class_imbalance.is_imbalanced ? 'critical' : 'good'}>
              {class_imbalance.majority_minority_ratio}:1 majority:minority
            </StatusBadge>
          )}
        </div>
        <div className="panel" style={{ padding: 16 }}>
          {class_imbalance.applicable ? (
            <ClassDistributionChart classDistribution={class_imbalance.class_distribution} />
          ) : (
            <p className="text-muted">{class_imbalance.reason}</p>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title"><h2>Multicollinearity</h2>
          {(() => {
            const pairCount = multicollinearity.high_correlation_pairs?.length || 0;
            const vifValues = Object.values(multicollinearity.vif || {});
            const threshold = multicollinearity.vif_threshold_used;
            const highVifCount = vifValues.filter((v) => v === null || v >= threshold).length;
            const flagged = pairCount > 0 || highVifCount > 0;
            return (
              <StatusBadge status={flagged ? 'warn' : 'good'}>
                {pairCount} correlated pair(s), {highVifCount} high-VIF feature(s)
              </StatusBadge>
            );
          })()}
        </div>
        <div className="panel" style={{ padding: 16 }}>
          {Object.keys(multicollinearity.vif || {}).length > 0 ? (
            <>
              <VifChart vif={multicollinearity.vif} threshold={multicollinearity.vif_threshold_used} />
              <p className="section-note">Dashed line marks the VIF threshold ({multicollinearity.vif_threshold_used}).</p>
            </>
          ) : (
            <p className="text-muted">{multicollinearity.note || 'Not enough numeric columns to evaluate.'}</p>
          )}
          {multicollinearity.high_correlation_pairs?.length > 0 && (
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
          <p className="section-note">{potential_leakage.disclaimer}</p>
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
            {!train_test_distribution_diff.applicable ? (
              <p className="text-muted">{train_test_distribution_diff.reason}</p>
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

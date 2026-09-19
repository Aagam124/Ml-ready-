import { useEffect, useState, useCallback } from 'react';
import { Outlet, useParams, Link } from 'react-router-dom';
import { getDataset } from '../lib/api';
import { useSelectedDataset } from '../lib/SelectedDatasetContext';
import { datasetStatusToLabel, formatBytes, formatDate } from '../lib/format';
import StatusBadge from './StatusBadge';

const STATUS_TO_BADGE = { uploaded: 'neutral', analyzed: 'warn', cleaned: 'good' };

export default function DatasetLayout() {
  const { id } = useParams();
  const { setSelected } = useSelectedDataset();
  const [dataset, setDataset] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(() => {
    setLoading(true);
    getDataset(id)
      .then((d) => {
        setDataset(d);
        setSelected(d);
        setError('');
      })
      .catch(() => setError('Could not load this dataset. It may have been deleted.'))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  if (loading) return null;

  if (error) {
    return (
      <div className="content">
        <p className="error-text">{error}</p>
        <Link to="/">← Back to Dashboard</Link>
      </div>
    );
  }

  return (
    <div className="content">
      <div className="page-header-row">
        <div>
          <h1>{dataset.filename}</h1>
          <p className="sub">
            Uploaded {formatDate(dataset.uploaded_at)} · {formatBytes(dataset.metadata?.size_bytes)}
          </p>
        </div>
        <StatusBadge status={STATUS_TO_BADGE[dataset.status] || 'neutral'}>
          {datasetStatusToLabel(dataset.status)}
        </StatusBadge>
      </div>

      <div className="data-strip">
        <div className="data-strip-item">
          <div className="data-strip-label">Rows</div>
          <div className="data-strip-value">{dataset.metadata?.n_rows ?? '—'}</div>
        </div>
        <div className="data-strip-item">
          <div className="data-strip-label">Columns</div>
          <div className="data-strip-value">{dataset.metadata?.n_columns ?? '—'}</div>
        </div>
        <div className="data-strip-item">
          <div className="data-strip-label">File size</div>
          <div className="data-strip-value small">{formatBytes(dataset.metadata?.size_bytes)}</div>
        </div>
        <div className="data-strip-item">
          <div className="data-strip-label">Status</div>
          <div className="data-strip-value small">{datasetStatusToLabel(dataset.status)}</div>
        </div>
      </div>

      <Outlet context={{ dataset, refetchDataset: refetch }} />
    </div>
  );
}

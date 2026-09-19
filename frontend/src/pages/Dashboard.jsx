import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { deleteDataset, listDatasets } from '../lib/api';
import { useSelectedDataset } from '../lib/SelectedDatasetContext';
import { datasetStatusToLabel, formatDate, formatNumber } from '../lib/format';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';

const STATUS_TO_BADGE = { uploaded: 'neutral', analyzed: 'warn', cleaned: 'good' };

export default function Dashboard() {
  const [datasets, setDatasets] = useState(null);
  const [error, setError] = useState('');
  const { setSelected } = useSelectedDataset();
  const navigate = useNavigate();

  function load() {
    listDatasets()
      .then(setDatasets)
      .catch(() => setError('Could not load your datasets.'));
  }

  useEffect(load, []);

  function openDataset(dataset) {
    setSelected(dataset);
    navigate(`/datasets/${dataset.id}`);
  }

  async function handleDelete(e, dataset) {
    e.stopPropagation();
    if (!window.confirm(`Delete "${dataset.filename}"? This can't be undone.`)) return;
    await deleteDataset(dataset.id);
    load();
  }

  const totalAnalyzed = datasets?.filter((d) => d.status !== 'uploaded').length ?? 0;

  return (
    <div className="content">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p className="sub">Every dataset you've uploaded, and where each one stands.</p>
      </div>

      {datasets && (
        <div className="data-strip">
          <div className="data-strip-item">
            <div className="data-strip-label">Datasets</div>
            <div className="data-strip-value">{datasets.length}</div>
          </div>
          <div className="data-strip-item">
            <div className="data-strip-label">Analyzed</div>
            <div className="data-strip-value">{totalAnalyzed}</div>
          </div>
        </div>
      )}

      {error && <p className="error-text">{error}</p>}

      {datasets && datasets.length === 0 && (
        <EmptyState
          title="No datasets yet"
          description="Upload a CSV or Excel file to get a readiness score, quality report, and ML risk analysis."
          action={<Link to="/upload"><button>Upload a dataset</button></Link>}
        />
      )}

      {datasets && datasets.length > 0 && (
        <div className="panel">
          {datasets.map((dataset) => (
            <div key={dataset.id} className="dataset-row" onClick={() => openDataset(dataset)} style={{ cursor: 'pointer' }}>
              <div className="dataset-row-main">
                <div className="dataset-row-name">{dataset.filename}</div>
                <div className="dataset-row-meta">
                  {formatNumber(dataset.metadata?.n_rows)} rows · {formatNumber(dataset.metadata?.n_columns)} cols · uploaded {formatDate(dataset.uploaded_at)}
                </div>
              </div>
              <div className="dataset-row-actions">
                <StatusBadge status={STATUS_TO_BADGE[dataset.status] || 'neutral'}>
                  {datasetStatusToLabel(dataset.status)}
                </StatusBadge>
                <button className="secondary" onClick={(e) => handleDelete(e, dataset)}>Delete</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

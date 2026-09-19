import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadDataset, apiErrorMessage } from '../lib/api';
import { useSelectedDataset } from '../lib/SelectedDatasetContext';

const ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls'];

export default function UploadDataset() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const { setSelected } = useSelectedDataset();

  function validateAndSetFile(candidate) {
    setError('');
    const ext = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type "${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`);
      return;
    }
    if (candidate.size === 0) {
      setError('That file is empty.');
      return;
    }
    setFile(candidate);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) validateAndSetFile(e.dataTransfer.files[0]);
  }

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const dataset = await uploadDataset(file, setProgress);
      setSelected(dataset);
      navigate(`/datasets/${dataset.id}`);
    } catch (err) {
      setError(apiErrorMessage(err, 'Upload failed.'));
      setUploading(false);
    }
  }

  return (
    <div className="content" style={{ maxWidth: 640 }}>
      <div className="page-header">
        <h1>Upload Dataset</h1>
        <p className="sub">CSV or Excel, up to 50 MB. Nothing is analyzed until you ask for it.</p>
      </div>

      <div
        className={`dropzone${dragActive ? ' active' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef} type="file" accept=".csv,.xlsx,.xls"
          onChange={(e) => e.target.files?.[0] && validateAndSetFile(e.target.files[0])}
        />
        {file ? (
          <div>
            <strong>{file.name}</strong>
            <div className="dropzone-hint">{(file.size / 1024).toFixed(0)} KB — click to choose a different file</div>
          </div>
        ) : (
          <div>
            <div>Drop a CSV or Excel file here, or click to browse</div>
            <div className="dropzone-hint">.csv, .xlsx, .xls</div>
          </div>
        )}
      </div>

      {error && <p className="error-text">{error}</p>}

      {uploading && (
        <div style={{ marginTop: 16 }}>
          <div className="hairline" style={{ position: 'relative', height: 4, borderTop: 'none', background: 'var(--line)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ width: `${progress}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.15s ease' }} />
          </div>
          <p className="section-note">Uploading… {progress}%</p>
        </div>
      )}

      <div style={{ marginTop: 20 }}>
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? 'Uploading…' : 'Upload'}
        </button>
      </div>
    </div>
  );
}

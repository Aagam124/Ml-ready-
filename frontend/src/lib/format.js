/** Maps a readiness-score category (from the backend's SCORE_BANDS) to a
 * status color class. Kept in one place so "what counts as good" is defined
 * exactly once. */
export function scoreCategoryToStatus(category) {
  if (category === 'Excellent' || category === 'Good') return 'good';
  if (category === 'Needs Improvement') return 'warn';
  return 'critical';
}

export function datasetStatusToLabel(status) {
  const map = {
    uploaded: 'Uploaded',
    analyzed: 'Analyzed',
    cleaned: 'Cleaned',
  };
  return map[status] || status;
}

export function formatNumber(value) {
  if (value === null || value === undefined) return '—';
  return new Intl.NumberFormat('en-US').format(value);
}

export function formatPercent(value) {
  if (value === null || value === undefined) return '—';
  return `${value}%`;
}

export function formatDate(isoString) {
  if (!isoString) return '—';
  const date = new Date(isoString);
  return date.toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

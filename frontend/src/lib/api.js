import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const client = axios.create({ baseURL: API_BASE_URL });

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('mlready_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('mlready_token');
      localStorage.removeItem('mlready_user');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  },
);

/** Pulls FastAPI's `{"detail": "..."}` shape into a plain message string. */
export function apiErrorMessage(error, fallback = 'Something went wrong. Please try again.') {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
  return fallback;
}

// ---- auth --------------------------------------------------------------

export const registerUser = (payload) => client.post('/auth/register', payload).then((r) => r.data);
export const loginUser = (payload) => client.post('/auth/login', payload).then((r) => r.data);
export const getMe = () => client.get('/auth/me').then((r) => r.data);

// ---- datasets ------------------------------------------------------------

export const uploadDataset = (file, onProgress) => {
  const form = new FormData();
  form.append('file', file);
  return client
    .post('/datasets/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress
        ? (evt) => onProgress(Math.round((evt.loaded / evt.total) * 100))
        : undefined,
    })
    .then((r) => r.data);
};

export const listDatasets = () => client.get('/datasets').then((r) => r.data);
export const getDataset = (id) => client.get(`/datasets/${id}`).then((r) => r.data);
export const deleteDataset = (id) => client.delete(`/datasets/${id}`).then((r) => r.data);

// ---- analysis ------------------------------------------------------------

export const analyzeDataset = (id, targetColumn) =>
  client.post(`/datasets/${id}/analyze`, { target_column: targetColumn || null }).then((r) => r.data);

export const getReport = (id) => client.get(`/datasets/${id}/report`).then((r) => r.data);
export const listReports = (id) => client.get(`/datasets/${id}/reports`).then((r) => r.data);
export const listCleaningHistory = (id) => client.get(`/datasets/${id}/cleaning-history`).then((r) => r.data);

export const cleanDataset = (id, operations) =>
  client.post(`/datasets/${id}/clean`, { operations }).then((r) => r.data);

// ---- models (Phase 5 - routes not built yet; wired ahead of time so the
// Model Comparison page just starts working once they exist) -------------

export const trainModel = (id, targetColumn) =>
  client.post(`/datasets/${id}/train`, { target_column: targetColumn }).then((r) => r.data);

export const getModelResults = (id) => client.get(`/datasets/${id}/model-results`).then((r) => r.data);
export const listModelResultsHistory = (id) => client.get(`/datasets/${id}/model-results/history`).then((r) => r.data);

// ---- reports (Phase 6) ---------------------------------------------------

export const getFullReport = (id) => client.get(`/datasets/${id}/full-report`).then((r) => r.data);

/** Fetches the export as a blob (auth requires a header, so a plain
 * navigation won't work) and triggers a normal browser download from it. */
export function downloadReport(id, format) {
  return client
    .get(`/datasets/${id}/report/export`, { params: { format }, responseType: 'blob' })
    .then((response) => {
      const disposition = response.headers['content-disposition'] || '';
      const match = disposition.match(/filename="?([^"]+)"?/);
      const filename = match ? match[1] : `report.${format}`;
      const blobUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(blobUrl);
    });
}

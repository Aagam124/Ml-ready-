import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider, ProtectedRoute } from './lib/AuthContext';
import { SelectedDatasetProvider } from './lib/SelectedDatasetContext';
import ErrorBoundary from './components/ErrorBoundary';
import AppShell from './components/AppShell';
import DatasetLayout from './components/DatasetLayout';

import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import UploadDataset from './pages/UploadDataset';
import DatasetOverview from './pages/DatasetOverview';
import DataQualityReport from './pages/DataQualityReport';
import MLRiskAnalysis from './pages/MLRiskAnalysis';
import Cleaning from './pages/Cleaning';
import ModelComparison from './pages/ModelComparison';
import FullReport from './pages/FullReport';
import AnalysisHistory from './pages/AnalysisHistory';
import NotFound from './pages/NotFound';

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <SelectedDatasetProvider>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />

              <Route
                element={
                  <ProtectedRoute>
                    <AppShell />
                  </ProtectedRoute>
                }
              >
                <Route path="/" element={<Dashboard />} />
                <Route path="/upload" element={<UploadDataset />} />

                <Route path="/datasets/:id" element={<DatasetLayout />}>
                  <Route index element={<DatasetOverview />} />
                  <Route path="quality" element={<DataQualityReport />} />
                  <Route path="risks" element={<MLRiskAnalysis />} />
                  <Route path="clean" element={<Cleaning />} />
                  <Route path="models" element={<ModelComparison />} />
                  <Route path="report" element={<FullReport />} />
                  <Route path="history" element={<AnalysisHistory />} />
                </Route>
              </Route>

              <Route path="*" element={<NotFound />} />
            </Routes>
          </SelectedDatasetProvider>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

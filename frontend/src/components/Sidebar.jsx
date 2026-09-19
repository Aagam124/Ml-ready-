import { NavLink } from 'react-router-dom';
import { useAuth } from '../lib/AuthContext';
import { useSelectedDataset } from '../lib/SelectedDatasetContext';

const PIPELINE_LINKS = [
  { to: (id) => `/datasets/${id}`, label: 'Overview', end: true },
  { to: (id) => `/datasets/${id}/quality`, label: 'Data Quality' },
  { to: (id) => `/datasets/${id}/risks`, label: 'ML Risk Analysis' },
  { to: (id) => `/datasets/${id}/clean`, label: 'Cleaning' },
  { to: (id) => `/datasets/${id}/models`, label: 'Model Comparison' },
  { to: (id) => `/datasets/${id}/report`, label: 'Full Report' },
  { to: (id) => `/datasets/${id}/history`, label: 'Analysis History' },
];

export default function Sidebar() {
  const { user, logout } = useAuth();
  const { selected } = useSelectedDataset();

  return (
    <nav className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-mark">MLReady</div>
        <div className="sidebar-brand-sub">Readiness &amp; Risk Analyzer</div>
      </div>

      <div className="sidebar-nav">
        <NavLink to="/" end className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
          Dashboard
        </NavLink>
        <NavLink to="/upload" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
          Upload Dataset
        </NavLink>
      </div>

      <div className="sidebar-section-label">
        {selected ? `Current: ${selected.filename}` : 'No dataset selected'}
      </div>
      <div className="sidebar-nav">
        {PIPELINE_LINKS.map((link) =>
          selected ? (
            <NavLink
              key={link.label}
              to={link.to(selected.id)}
              end={link.end}
              className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
            >
              {link.label}
            </NavLink>
          ) : (
            <span key={link.label} className="sidebar-link disabled">
              {link.label}
            </span>
          ),
        )}
      </div>

      <div className="sidebar-footer">
        <div className="sidebar-user">{user?.username}</div>
        <button className="secondary" onClick={logout} style={{ width: '100%' }}>
          Log out
        </button>
      </div>
    </nav>
  );
}

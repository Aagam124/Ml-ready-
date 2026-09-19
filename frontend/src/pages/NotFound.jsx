import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="content">
      <div className="empty-state" style={{ marginTop: 40 }}>
        <h3>Page not found</h3>
        <p>That page doesn't exist.</p>
        <Link to="/"><button>Back to Dashboard</button></Link>
      </div>
    </div>
  );
}

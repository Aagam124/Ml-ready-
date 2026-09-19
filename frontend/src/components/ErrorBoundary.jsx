import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error('MLReady crashed:', error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="auth-shell">
          <div className="auth-card panel">
            <div className="auth-mark">Something went wrong</div>
            <p className="auth-sub">
              MLReady hit an unexpected error. Reloading usually fixes it — if it keeps happening, the browser
              console has the details.
            </p>
            <button style={{ width: '100%' }} onClick={() => window.location.assign('/')}>
              Back to Dashboard
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

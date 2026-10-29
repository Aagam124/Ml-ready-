import { useEffect, useState } from "react";
import { api, getBaseUrl, session } from "../api";
import { ErrorBox } from "../components/ui";
import type { ScreenProps } from "../flow";

interface Me { email: string | null; guest: boolean; expires_at?: string }

export function Settings({ flow, setFlow, go, theme, setTheme }: ScreenProps & { theme: string; setTheme: (t: string) => void }) {
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [confirm, setConfirm] = useState(false);
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem("mlready.api_url") || "");
  const [savedUrl, setSavedUrl] = useState(false);

  useEffect(() => {
    api<Me>("/auth/me").then(setMe).catch(setError);
  }, []);

  async function wipe() {
    try {
      await api(`/datasets/${flow.dataset!.dataset.id}`, { method: "DELETE" });
      setFlow({});
      go("landing");
    } catch (e) {
      setError(e);
    }
  }

  function saveApiUrl() {
    if (apiUrl.trim()) {
      localStorage.setItem("mlready.api_url", apiUrl.trim());
    } else {
      localStorage.removeItem("mlready.api_url");
    }
    setSavedUrl(true);
    setTimeout(() => {
      window.location.reload();
    }, 600);
  }

  return (
    <>
      <h1 style={{ fontSize: "1.9rem" }}>Settings</h1>
      {error !== null && <ErrorBox error={error} />}

      <div className="card">
        <h3>Backend API Connection</h3>
        <p className="muted" style={{ fontSize: "0.9rem" }}>
          Current API Endpoint: <code>{getBaseUrl()}</code>
        </p>
        <div style={{ marginTop: "0.5rem" }}>
          <label style={{ display: "block", fontSize: "0.85rem", marginBottom: "0.3rem" }}>
            Custom Backend URL (optional):
          </label>
          <div style={{ display: "flex", gap: "0.5rem", maxWidth: "450px" }}>
            <input
              type="text"
              placeholder="e.g. http://localhost:8000"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              style={{ flex: 1, padding: "0.5rem", borderRadius: "4px" }}
            />
            <button className="btn primary small" onClick={saveApiUrl}>
              {savedUrl ? "Saved! Reloading…" : "Save"}
            </button>
          </div>
          <p className="muted" style={{ fontSize: "0.8rem", marginTop: "0.3rem" }}>
            Leave blank to use the default relative <code>/api/v1</code> endpoint.
          </p>
        </div>
      </div>

      <div className="card">
        <h3>Account</h3>
        <p>
          {me ? (
            me.guest ? (
              <>
                Demo session. Everything you create is deleted automatically
                {me.expires_at ? ` on ${new Date(me.expires_at).toLocaleString()}` : ""}. Create an account on the home screen to keep your work.
              </>
            ) : (
              <>Signed in as <b>{me.email}</b>.</>
            )
          ) : (
            "Not connected to backend."
          )}
        </p>
        <button
          className="btn"
          onClick={() => {
            session.set(null);
            setFlow({});
            go("landing");
          }}
        >
          Sign out
        </button>
      </div>

      <div className="card">
        <h3>Appearance</h3>
        <label htmlFor="th">Theme</label>
        <select
          id="th"
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          style={{ maxWidth: 220 }}
        >
          <option value="auto">Match my device</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>

      {flow.dataset && (
        <div className="card">
          <h3>Delete everything for “{flow.dataset.dataset.name}”</h3>
          <p className="muted">
            Removes the dataset, every version, analyses, findings, plans, model runs, reports and share links. This cannot be undone.
          </p>
          {!confirm ? (
            <button className="btn" onClick={() => setConfirm(true)}>
              Delete everything…
            </button>
          ) : (
            <div className="row">
              <strong>Are you sure?</strong>
              <button
                className="btn"
                style={{ borderColor: "var(--bad)", color: "var(--bad)" }}
                onClick={() => void wipe()}
              >
                Yes, delete permanently
              </button>
              <button className="btn" onClick={() => setConfirm(false)}>
                Cancel
              </button>
            </div>
          )}
        </div>
      )}
    </>
  );
}

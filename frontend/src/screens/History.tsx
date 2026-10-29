import { useEffect, useState } from "react";
import { api, post } from "../api";
import { Empty, ErrorBox, Skeleton } from "../components/ui";
import type { ScreenProps } from "../flow";
import type { Analysis, DatasetSummary, Diff, Version } from "../types";

interface AuditEvent {
  id: string;
  action: string;
  object_ref: string;
  created_at: string;
}

export function History({ flow, setFlow, go }: ScreenProps) {
  const ds = flow.dataset;
  const [versions, setVersions] = useState<Version[] | null>(null);
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditEvent[]>([]);
  const [error, setError] = useState<unknown>(null);

  // Comparison state
  const [compA, setCompA] = useState<string>("");
  const [compB, setCompB] = useState<string>("");
  const [diff, setDiff] = useState<Diff | null>(null);
  const [diffLoading, setDiffLoading] = useState<boolean>(false);

  const load = () => {
    if (!ds) return;
    api<{ items: Version[] }>(`/datasets/${ds.dataset.id}/versions`)
      .then((r) => setVersions(r.items))
      .catch(setError);
    api<{ items: Analysis[] }>(`/datasets/${ds.dataset.id}/analyses`)
      .then((r) => {
        setAnalyses(r.items);
        if (r.items.length >= 2) {
          setCompA(r.items[r.items.length - 2].id);
          setCompB(r.items[r.items.length - 1].id);
        }
      })
      .catch(() => undefined);
    api<{ items: AuditEvent[] }>(`/workspaces/audit-log?limit=10`)
      .then((r) => setAuditLogs(r.items))
      .catch(() => undefined);
  };

  useEffect(load, [ds?.dataset.id, ds?.dataset.current_version_id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!ds) return <Empty title="No dataset yet"><button className="btn primary" onClick={() => go("landing")}>Add one</button></Empty>;

  async function select(id: string) {
    try {
      await post(`/datasets/${ds!.dataset.id}/versions/${id}/select`);
      const d = await api<DatasetSummary>(`/datasets/${ds!.dataset.id}`);
      setFlow({ ...flow, dataset: d, analysis: undefined, plan: undefined, run: undefined, afterAnalysis: undefined });
      load();
    } catch (e) {
      setError(e);
    }
  }

  async function compareAnalyses() {
    if (!compA || !compB || compA === compB) return;
    setDiffLoading(true);
    setDiff(null);
    try {
      const res = await api<Diff>(`/analyses/compare/${compA}/${compB}`);
      setDiff(res);
    } catch (e) {
      setError(e);
    } finally {
      setDiffLoading(false);
    }
  }

  return (
    <>
      <h1 style={{ fontSize: "1.9rem" }}>Dataset History & Versions</h1>
      <p className="muted">
        Every transformation or clean produces an immutable version. Pick any version to restore.
      </p>
      {error !== null && <ErrorBox error={error} />}

      {!versions ? (
        <Skeleton lines={3} height={36} />
      ) : (
        <ol className="timeline">
          {versions.map((v) => (
            <li key={v.id}>
              <strong>{v.parent_version_id ? "Cleaned version" : "Original upload"}</strong>{" "}
              {v.id === ds.dataset.current_version_id && <span className="badge sev-info">current</span>}
              <div className="muted">
                {new Date(v.created_at).toLocaleString()} · {v.n_rows.toLocaleString()} rows × {v.n_cols} columns
                {v.created_by_step ? ` · ${v.created_by_step.startsWith("cleaning_plan") ? "from cleaning plan" : v.created_by_step}` : ""}
              </div>
              {v.id !== ds.dataset.current_version_id && (
                <button className="btn small" onClick={() => void select(v.id)}>
                  Make this current
                </button>
              )}
            </li>
          ))}
        </ol>
      )}

      <h2>Analyses & Runs</h2>
      {analyses.length === 0 ? (
        <p className="muted">No analyses executed yet.</p>
      ) : (
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Target</th>
                <th>Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {analyses.map((a) => (
                <tr key={a.id}>
                  <td>{(a as unknown as { created_at: string }).created_at?.slice(0, 16).replace("T", " ")}</td>
                  <td>{a.target_column ?? "none"}</td>
                  <td className="num">
                    {a.score ? a.score.value.toFixed(1) : "…"}
                    {a.score?.provisional ? " (provisional)" : ""}
                  </td>
                  <td>
                    <span className={`badge ${a.status === "done" ? "sev-low" : "sev-info"}`}>{a.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {analyses.length >= 2 && (
        <div style={{ marginTop: "2rem", padding: "1.2rem", background: "var(--card-bg, #1e293b)", borderRadius: "8px", border: "1px solid var(--border-color, #334155)" }}>
          <h3>Compare Two Analyses</h3>
          <p className="muted" style={{ fontSize: "0.9rem" }}>
            Measure what improved, what regressed, and score delta between any two runs.
          </p>
          <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap", marginBottom: "1rem" }}>
            <div>
              <label style={{ display: "block", fontSize: "0.8rem", marginBottom: "0.2rem" }}>Baseline (Before)</label>
              <select value={compA} onChange={(e) => setCompA(e.target.value)} style={{ padding: "0.4rem", borderRadius: "4px" }}>
                {analyses.map((a, i) => (
                  <option key={a.id} value={a.id}>
                    Run #{i + 1} ({a.score ? `${a.score.value.toFixed(1)} pts` : a.status})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ display: "block", fontSize: "0.8rem", marginBottom: "0.2rem" }}>Target (After)</label>
              <select value={compB} onChange={(e) => setCompB(e.target.value)} style={{ padding: "0.4rem", borderRadius: "4px" }}>
                {analyses.map((a, i) => (
                  <option key={a.id} value={a.id}>
                    Run #{i + 1} ({a.score ? `${a.score.value.toFixed(1)} pts` : a.status})
                  </option>
                ))}
              </select>
            </div>
            <button className="btn primary small" onClick={() => void compareAnalyses()} disabled={diffLoading || compA === compB}>
              {diffLoading ? "Comparing…" : "Compare Runs"}
            </button>
          </div>

          {diff && (
            <div style={{ marginTop: "1rem", borderTop: "1px solid var(--border-color, #334155)", paddingTop: "1rem" }}>
              <h4>Comparison Results</h4>
              <div style={{ display: "flex", gap: "2rem", marginBottom: "1rem" }}>
                <div>
                  <div className="muted" style={{ fontSize: "0.8rem" }}>Score Delta</div>
                  <strong style={{ fontSize: "1.4rem", color: (diff.score_delta ?? 0) >= 0 ? "#10b981" : "#ef4444" }}>
                    {(diff.score_delta ?? 0) > 0 ? `+${diff.score_delta}` : (diff.score_delta ?? "0")}
                  </strong>
                </div>
                <div>
                  <div className="muted" style={{ fontSize: "0.8rem" }}>Resolved Issues</div>
                  <strong style={{ fontSize: "1.4rem", color: "#10b981" }}>{diff.resolved.length}</strong>
                </div>
                <div>
                  <div className="muted" style={{ fontSize: "0.8rem" }}>New Regressions</div>
                  <strong style={{ fontSize: "1.4rem", color: diff.new.length > 0 ? "#ef4444" : "#10b981" }}>{diff.new.length}</strong>
                </div>
                <div>
                  <div className="muted" style={{ fontSize: "0.8rem" }}>Unchanged Checks</div>
                  <strong style={{ fontSize: "1.4rem" }}>{diff.unchanged_count}</strong>
                </div>
              </div>

              {diff.resolved.length > 0 && (
                <div style={{ marginBottom: "0.8rem" }}>
                  <div style={{ fontWeight: 600, color: "#10b981", fontSize: "0.85rem", marginBottom: "0.4rem" }}>Resolved:</div>
                  <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.85rem" }}>
                    {diff.resolved.map((r, i) => (
                      <li key={i}>{r.plain_text} (+{r.points} pts)</li>
                    ))}
                  </ul>
                </div>
              )}

              {diff.new.length > 0 && (
                <div>
                  <div style={{ fontWeight: 600, color: "#ef4444", fontSize: "0.85rem", marginBottom: "0.4rem" }}>New Regressions:</div>
                  <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.85rem" }}>
                    {diff.new.map((n, i) => (
                      <li key={i}>{n.plain_text} (-{n.points} pts)</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {auditLogs.length > 0 && (
        <div style={{ marginTop: "2.5rem" }}>
          <h2>Workspace Audit Trail (Governance)</h2>
          <p className="muted" style={{ fontSize: "0.85rem" }}>
            Immutable trace of dataset modifications, cleaning operations, and workspace actions.
          </p>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Action</th>
                  <th>Resource Ref</th>
                </tr>
              </thead>
              <tbody>
                {auditLogs.map((log, idx) => (
                  <tr key={idx}>
                    <td style={{ fontSize: "0.85rem" }}>{new Date(log.created_at).toLocaleString()}</td>
                    <td><code>{log.action}</code></td>
                    <td style={{ fontSize: "0.85rem", color: "var(--text-muted, #94a3b8)" }}>{log.object_ref}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

import { useState } from "react";
import { download, post } from "../api";
import { Empty, ErrorBox } from "../components/ui";
import type { ScreenProps } from "../flow";

export function Export({ flow, go }: ScreenProps) {
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [share, setShare] = useState<string | null>(null);
  const [lastReport, setLastReport] = useState<string | null>(null);
  const a = flow.analysis;
  if (!a || a.status !== "done") return <Empty title="Run an analysis first"><button className="btn primary" onClick={() => go("landing")}>Start</button></Empty>;

  async function report(format: "html" | "pdf") {
    setError(null); setBusy(format);
    try {
      const rep = await post<{ id: string }>(`/analyses/${a!.id}/reports`, { format, model_run_id: flow.run?.id });
      setLastReport(rep.id);
      await download(`/reports/${rep.id}/download`, `mlready-report.${format}`);
    } catch (e) { setError(e); } finally { setBusy(null); }
  }
  async function makeLink() {
    setError(null);
    try {
      let id = lastReport;
      if (!id) { id = (await post<{ id: string }>(`/analyses/${a!.id}/reports`, { format: "html", model_run_id: flow.run?.id })).id; setLastReport(id); }
      const s = await post<{ share_url_path: string; expires_at: string }>(`/reports/${id}/share`, { days: 7 });
      setShare(`${location.origin}${s.share_url_path}`);
    } catch (e) { setError(e); }
  }
  const dsId = flow.dataset!.dataset.id;
  return (
    <>
      <h1 style={{ fontSize: "1.9rem" }}>Export</h1>
      <p className="muted">Take everything with you: the report, the cleaned data and code that reproduces it.</p>
      {error !== null && <ErrorBox error={error} />}
      <div className="grid3">
        <div className="card"><h3>Report</h3><p className="muted">Score, every finding with evidence{flow.run?.result ? ", and the before/after comparison" : ""}.</p>
          <div className="row"><button className="btn primary" disabled={busy !== null} onClick={() => void report("html")}>{busy === "html" ? "Preparing" : "HTML"}</button><button className="btn" disabled={busy !== null} onClick={() => void report("pdf")}>{busy === "pdf" ? "Preparing" : "PDF"}</button></div></div>
        <div className="card"><h3>Cleaned data</h3><p className="muted">{flow.plan?.status === "applied" ? "The current version, as CSV." : "Apply a cleaning plan to get a cleaned version. The original is available too."}</p>
          <button className="btn" onClick={() => void download(`/datasets/${dsId}/download`, flow.dataset!.version.original_filename)}>Download CSV</button></div>
        <div className="card"><h3>Code</h3><p className="muted">A commented script and a scikit-learn Pipeline that reproduce the cleaning exactly.</p>
          <div className="row">{flow.plan ? <><button className="btn" onClick={() => void download(`/plans/${flow.plan!.id}/export?kind=script`, "clean_data.py")}>Script</button><button className="btn" onClick={() => void download(`/plans/${flow.plan!.id}/export?kind=pipeline`, "mlready_pipeline.py")}>Pipeline</button></> : <button className="btn" onClick={() => go("plan")}>Build a plan</button>}</div></div>
      </div>
      <h2>Share</h2>
      <div className="card"><p className="muted">Anyone with the link can read the HTML report for 7 days. They cannot see your data or sign in.</p>
        <button className="btn" onClick={() => void makeLink()}>Create 7-day link</button>
        {share && <p><input type="text" readOnly value={share} aria-label="Share link" onFocus={(e) => e.currentTarget.select()} /></p>}</div>
    </>
  );
}

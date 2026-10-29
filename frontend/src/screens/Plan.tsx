import { useCallback, useEffect, useState } from "react";
import { api, download, patch, post, watchJob } from "../api";
import { Empty, ErrorBox, ProgressBar, Rich, SeverityBadge, Skeleton } from "../components/ui";
import type { ScreenProps } from "../flow";
import type { Analysis, DatasetSummary, Job, Plan as PlanT, Preview, Step } from "../types";

function Sample({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return <p className="muted">No rows to show.</p>;
  const cols = Object.keys(rows[0]);
  return <div className="scroll"><table><thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c} className="num">{r[c] === null || r[c] === undefined ? <span className="muted">missing</span> : String(r[c])}</td>)}</tr>)}</tbody></table></div>;
}

export function Plan({ flow, setFlow, go }: ScreenProps) {
  const analysis = flow.analysis!;
  const [plan, setPlan] = useState<PlanT | null>(flow.plan && flow.plan.analysis_id === analysis.id ? flow.plan : null);
  const [error, setError] = useState<unknown>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [previews, setPreviews] = useState<Record<string, Preview | "loading">>({});
  const [after, setAfter] = useState<Analysis | null>(flow.afterAnalysis ?? null);

  useEffect(() => {
    if (plan) return;
    post<PlanT>(`/analyses/${analysis.id}/plan`).then((p) => { setPlan(p); setFlow({ ...flow, plan: p }); }).catch(setError);
  }, [analysis.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = useCallback(async (s: Step) => {
    if (!plan) return;
    setPlan({ ...plan, steps: plan.steps.map((x) => (x.step_id === s.step_id ? { ...x, enabled: !x.enabled } : x)) });   // optimistic
    try { const p = await patch<PlanT>(`/plans/${plan.id}`, { steps: [{ step_id: s.step_id, enabled: !s.enabled }] }); setPlan(p); setPreviews({}); }
    catch (e) { setError(e); setPlan(plan); }
  }, [plan]);

  async function preview(s: Step) {
    if (!plan) return;
    setPreviews((p) => ({ ...p, [s.step_id]: "loading" }));
    try { const r = await post<Preview>(`/plans/${plan.id}/steps/${s.step_id}/preview`); setPreviews((p) => ({ ...p, [s.step_id]: r })); }
    catch (e) { setError(e); setPreviews((p) => { const n = { ...p }; delete n[s.step_id]; return n; }); }
  }

  async function apply() {
    if (!plan) return;
    setError(null); setBusy("Applying your plan");
    try {
      const { job_id } = await post<{ job_id: string }>(`/plans/${plan.id}/apply`);
      await watchJob(job_id, setJob);
      const done = await api<PlanT>(`/plans/${plan.id}`);
      setBusy("Re-checking the cleaned data");
      const next = await api<Analysis>(`/analyses/${done.result_analysis_id}`);
      if (next.job_id) await watchJob(next.job_id, setJob);
      const finished = await api<Analysis>(`/analyses/${done.result_analysis_id}`);
      const ds = await api<DatasetSummary>(`/datasets/${flow.dataset!.dataset.id}`);
      setPlan(done); setAfter(finished); setFlow({ ...flow, plan: done, afterAnalysis: finished, dataset: ds });
    } catch (e) { setError(e); } finally { setBusy(null); }
  }

  async function undo() {
    try { const ds0 = flow.dataset!.dataset.id; await post(`/datasets/${ds0}/undo`); const ds = await api<DatasetSummary>(`/datasets/${ds0}`);
      setAfter(null); setPlan(null); setFlow({ ...flow, dataset: ds, plan: undefined, afterAnalysis: undefined, run: undefined }); go("verdict"); }
    catch (e) { setError(e); }
  }

  if (busy) return <ProgressBar job={job} label={busy} />;
  const enabled = plan?.steps.filter((s) => s.enabled).length ?? 0;
  const applied = plan?.status === "applied";
  return (
    <>
      <h1 style={{ fontSize: "1.9rem" }}>Cleaning plan</h1>
      <p className="muted">Each step fixes one finding. Preview it on a sample, switch it off if you disagree, then apply. Your original file is never changed; applying creates a new version you can undo.</p>
      {error !== null && <ErrorBox error={error} />}
      {applied && after?.score && analysis.score && (
        <div className="callout good" role="status"><strong>Applied.</strong> Readiness score {analysis.score.value.toFixed(0)} → <strong>{after.score.value.toFixed(0)}</strong> ({after.score.band}).
          <div className="row" style={{ marginTop: 8 }}><button className="btn primary" onClick={() => go("results")}>Measure the effect on real models</button><button className="btn" onClick={() => void undo()}>Undo (go back to the previous version)</button></div></div>)}
      {!plan ? <Skeleton lines={5} height={46} /> : plan.steps.length === 0 ? <Empty title="Nothing to clean">No automatic fixes were needed for this dataset.</Empty> : (
        <>
          {plan.steps.map((s) => {
            const pv = previews[s.step_id];
            return (
              <div key={s.step_id} className={`step${s.enabled ? "" : " off"}`}>
                <input className="switch" type="checkbox" checked={s.enabled} disabled={applied} onChange={() => void toggle(s)} aria-label={`Enable step: ${s.label}`} />
                <div>
                  <div className="row" style={{ gap: 8 }}>{s.severity && <SeverityBadge s={s.severity} />}<strong>{s.label}</strong></div>
                  <p className="muted" style={{ margin: "2px 0" }}>{s.columns.length ? `Columns: ${s.columns.join(", ")}` : "Whole dataset"}{s.finding_code ? ` · fixes ${s.finding_code.replace(/_/g, " ")}` : ""}{!s.enabled && s.finding_code === "target_leakage" ? " · off by default because the evidence is not conclusive" : ""}</p>
                  {pv === "loading" && <Skeleton lines={2} />}
                  {pv && pv !== "loading" && (
                    <div className="card" style={{ marginTop: 8 }}>
                      <p style={{ margin: 0 }}>{pv.rows_removed ? `${pv.rows_removed} rows removed` : "No rows removed"} · {pv.cells_changed} cells changed · columns: {pv.columns_affected.join(", ") || "none"} <span className="muted">({pv.denominator})</span></p>
                      <div className="grid3"><div><b>Before</b><Sample rows={pv.before_sample} /></div><div><b>After</b><Sample rows={pv.after_sample} /></div></div>
                    </div>)}
                </div>
                <div className="actions"><button className="btn small" onClick={() => void preview(s)} disabled={pv === "loading"}>Preview</button></div>
              </div>);
          })}
          <div className="row" style={{ margin: "18px 0" }}>
            {!applied && <button className="btn primary big" disabled={enabled === 0} onClick={() => void apply()}>Apply {enabled} step{enabled === 1 ? "" : "s"}</button>}
            <button className="btn" onClick={() => void download(`/plans/${plan.id}/export?kind=script`, "clean_data.py")}>Download Python script</button>
            <button className="btn" onClick={() => void download(`/plans/${plan.id}/export?kind=pipeline`, "mlready_pipeline.py")}>Download scikit-learn Pipeline</button>
          </div>
          <p className="muted"><Rich text="The exported code learns imputation, scaling and encoding only from training rows, so it never leaks test information." /></p>
        </>)}
    </>
  );
}

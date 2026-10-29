import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, post, watchJob } from "../api";
import { Empty, ErrorBox, ProgressBar, Rich, fmt } from "../components/ui";
import type { ScreenProps } from "../flow";
import type { Diff, Job, ModelRun } from "../types";

export function Results({ flow, setFlow, go }: ScreenProps) {
  const plan = flow.plan;
  const [run, setRun] = useState<ModelRun | null>(flow.run ?? null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [diff, setDiff] = useState<Diff | null>(null);

  useEffect(() => {
    if (flow.analysis && flow.afterAnalysis?.status === "done") api<Diff>(`/analyses/compare/${flow.analysis.id}/${flow.afterAnalysis.id}`).then(setDiff).catch(() => undefined);
  }, [flow.analysis, flow.afterAnalysis]);

  async function start() {
    if (!plan) return;
    setError(null); setBusy(true);
    try {
      const r = await post<{ model_run_id: string; job_id: string }>("/model-runs", { plan_id: plan.id });
      await watchJob(r.job_id, setJob);
      const res = await api<ModelRun>(`/model-runs/${r.model_run_id}`);
      setRun(res); setFlow({ ...flow, run: res });
    } catch (e) { setError(e); } finally { setBusy(false); }
  }

  if (!plan) return <Empty title="Build a cleaning plan first"><button className="btn primary" onClick={() => go("plan")}>Go to the cleaning plan</button></Empty>;
  if (busy) return <ProgressBar job={job} label="Training baseline models before and after" />;

  const cmp = run?.result?.comparison;
  const arms = run?.result?.arms;
  const chart = cmp && arms ? cmp.per_model.map((p) => {
    const b = arms.before.models.find((m) => m.name === p.name)!.cv[cmp.primary_metric];
    const a = arms.after.models.find((m) => m.name === p.name)!.cv[cmp.primary_metric];
    const err = (i: typeof b) => [i.mean !== null && i.ci_low !== null ? i.mean - i.ci_low : 0, i.mean !== null && i.ci_high !== null ? i.ci_high - i.mean : 0];
    return { name: p.name.replace(/_/g, " "), Before: b.mean, After: a.mean, bErr: err(b), aErr: err(a) };
  }) : [];

  return (
    <>
      <h1 style={{ fontSize: "1.9rem" }}>Did it actually help?</h1>
      <p className="muted">MLReady trains three baseline models on the original and the cleaned data using the <b>same split and the same cross-validation folds</b>, so the comparison is fair.</p>
      {error !== null && <ErrorBox error={error} onRetry={() => void start()} />}
      {diff && (
        <div className="card"><h3>What changed?</h3>
          <p>Score {diff.score_from?.toFixed(0)} → <b>{diff.score_to?.toFixed(0)}</b> ({diff.score_delta !== null && diff.score_delta >= 0 ? "+" : ""}{diff.score_delta?.toFixed(1)}).</p>
          {diff.resolved.length > 0 && <><b>Resolved ({diff.resolved.length})</b><ul>{diff.resolved.map((f) => <li key={f.code + f.columns.join()}>{f.code.replace(/_/g, " ")}{f.columns.length ? ` (${f.columns.join(", ")})` : ""} <span className="muted">+{f.points.toFixed(1)}</span></li>)}</ul></>}
          {diff.new.length > 0 && <><b>New ({diff.new.length})</b><ul>{diff.new.map((f) => <li key={f.code + f.columns.join()}>{f.plain_text}</li>)}</ul></>}
          <p className="muted">{diff.unchanged_count} findings are unchanged.</p></div>)}
      {!run && <div className="row"><button className="btn primary big" onClick={() => void start()}>Measure before vs after</button><span className="muted">Takes about a minute.</span></div>}
      {cmp && run?.result && (
        <>
          <div className={`callout ${cmp.verdict === "improved" ? "good" : cmp.verdict === "no_meaningful_change" ? "" : "warn"}`} role="status"><strong><Rich text={cmp.headline} /></strong></div>
          {cmp.notes.map((n) => <div className="callout warn" key={n}>{n}</div>)}
          <div className="card">
            <h3>{cmp.primary_metric.replace(/_/g, " ")} by model <span className="muted">(mean and 95% interval over {run.result.n_folds} folds)</span></h3>
            <div role="img" aria-label={`Bar chart of ${cmp.primary_metric} before and after for each model`} style={{ height: 300 }}>
              <ResponsiveContainer><BarChart data={chart}><CartesianGrid strokeDasharray="3 3" stroke="var(--line)" /><XAxis dataKey="name" tick={{ fill: "var(--fg)", fontSize: 12 }} /><YAxis domain={["auto", "auto"]} tick={{ fill: "var(--muted)", fontSize: 12 }} /><Tooltip /><Legend />
                <Bar dataKey="Before" fill="#8a94a6"><ErrorBar dataKey="bErr" stroke="var(--fg)" /></Bar>
                <Bar dataKey="After" fill="var(--accent)"><ErrorBar dataKey="aErr" stroke="var(--fg)" /></Bar></BarChart></ResponsiveContainer>
            </div>
            <div className="scroll"><table><thead><tr><th>Model</th><th>Before</th><th>After</th><th>Change</th><th>Paired 95% interval</th><th>Verdict</th></tr></thead><tbody>
              {cmp.per_model.map((p) => <tr key={p.name}><td>{p.name.replace(/_/g, " ")}</td><td className="num">{fmt(p.before)}</td><td className="num">{fmt(p.after)}</td><td className="num">{p.delta === null ? "n/a" : (p.delta >= 0 ? "+" : "") + p.delta.toFixed(3)}</td>
                <td className="num">{p.paired.ci_low === null ? "n/a" : `${fmt(p.paired.ci_low)} to ${fmt(p.paired.ci_high)}`}</td><td>{p.verdict.replace(/_/g, " ")}</td></tr>)}</tbody></table></div>
            <p className="muted">Scored on {run.result.common_test_rows.toLocaleString()} held-out rows shared by every run. {run.result.notes[1]}</p>
          </div>
          <div className="row"><button className="btn primary" onClick={() => go("export")}>Export report and code</button><button className="btn" onClick={() => void start()}>Run again</button></div>
        </>)}
    </>
  );
}

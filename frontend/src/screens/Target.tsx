import { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorBox, ProgressBar, Skeleton } from "../components/ui";
import { runAnalysis, ScreenProps } from "../flow";
import type { Candidate, Job } from "../types";

export function Target({ flow, setFlow, go }: ScreenProps) {
  const ds = flow.dataset!;
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [chosen, setChosen] = useState<string>("");
  const [job, setJob] = useState<Job | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api<{ candidates: Candidate[] }>(`/datasets/${ds.dataset.id}/target-suggestions`).then((r) => { setCands(r.candidates); setChosen(r.candidates[0]?.column ?? ""); }).catch(setError);
    api<{ profile: { columns: { name: string }[] } }>(`/datasets/${ds.dataset.id}/profile`).then((r) => setColumns(r.profile.columns.map((c) => c.name))).catch(() => undefined);
  }, [ds.dataset.id]);

  async function start(target: string | null) {
    setError(null); setRunning(true); setJob(null);
    try { const analysis = await runAnalysis(ds.dataset.id, target, setJob); setFlow({ ...flow, analysis, plan: undefined, run: undefined, afterAnalysis: undefined }); go("verdict"); }
    catch (e) { setError(e); setRunning(false); }
  }

  if (running) return <ProgressBar job={job} label="Analysing your data" />;
  const top = cands?.find((c) => c.column === chosen);
  return (
    <>
      <h1>What do you want to predict?</h1>
      <p className="muted">{ds.overview.n_rows.toLocaleString()} rows, {ds.overview.n_cols} columns. Pick the column your model should learn to predict (the target), so MLReady can check for imbalance and leakage.</p>
      {error !== null && <ErrorBox error={error} />}
      {!cands ? <Skeleton lines={4} height={22} /> : (
        <div className="card">
          {top ? (
            <>
              <div className="row between"><div><span className="muted">Best guess</span><h2 style={{ margin: 0 }}>{top.column}</h2></div>
                <span className="muted">{Math.round(top.confidence * 100)}% confident &middot; {top.task_type === "regression" ? "predicting a number" : top.task_type === "binary" ? "yes / no" : "several categories"}</span></div>
              <ul>{top.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
              <button className="btn primary big" onClick={() => void start(top.column)}>Yes, predict {top.column}</button>
            </>
          ) : <p>No obvious target column found. You can still run the checks that do not need one.</p>}
          <hr style={{ border: 0, borderTop: "1px solid var(--line)", margin: "18px 0" }} />
          <label htmlFor="tgt">Or choose another column</label>
          <div className="row">
            <select id="tgt" value={chosen} onChange={(e) => setChosen(e.target.value)} style={{ maxWidth: 320 }}>
              {(columns.length ? columns : cands.map((c) => c.column)).map((c) => <option key={c}>{c}</option>)}
            </select>
            <button className="btn" onClick={() => void start(chosen)}>Use this column</button>
            <button className="btn" onClick={() => void start(null)}>Skip for now</button>
          </div>
          <p className="muted">Skipping gives a provisional score: imbalance, leakage and model checks are switched off until you choose a target.</p>
        </div>
      )}
    </>
  );
}

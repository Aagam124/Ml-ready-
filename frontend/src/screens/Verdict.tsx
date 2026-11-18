import { Rich } from "../components/ui";
import { ScoreDial } from "../components/ScoreDial";
import { Waterfall } from "../components/Waterfall";
import type { ScreenProps } from "../flow";

export function Verdict({ flow, go }: ScreenProps) {
  const a = flow.analysis!;
  if (a.status === "failed") return (
    <div className="callout danger" role="alert">
      <strong>The analysis failed.</strong><p>{a.error}</p>
      {a.check_errors?.map((e) => <p key={e.check}><b>{e.check}:</b> {e.error}</p>)}
      <button className="btn" onClick={() => go("target")}>Go back and try again</button>
    </div>
  );
  const s = a.score!, sum = a.summary!;
  return (
    <>
      {a.status === "partial" && (
        <div className="callout danger" role="alert">
          <strong>Partial analysis — this is not a fully validated readiness result.</strong>
          <p>{a.check_errors?.length ?? 0} check(s) could not run. The score and findings may omit issues those checks would have detected.</p>
          {a.check_errors?.map((e) => <p key={e.check}><b>{e.check}:</b> {e.error}</p>)}
        </div>
      )}
      <div className="card row" style={{ gap: 28 }}>
        <ScoreDial value={s.value} band={s.band} provisional={s.provisional} />
        <div className="grow">
          <h1 style={{ fontSize: "1.9rem" }}>{sum.headline}</h1>
          <p className="muted">{a.target_column ? <>Predicting <b>{a.target_column}</b>.</> : "No target selected."} Scoring v{s.version}. Re-running gives the same result.</p>
          <div className="row">{(["critical", "high", "medium", "low"] as const).filter((k) => sum.counts[k]).map((k) => <span key={k} className={`badge sev-${k}`}>{sum.counts[k]} {k}</span>)}</div>
        </div>
      </div>
      {s.dimensions && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 style={{ margin: "0 0 10px 0", fontSize: "1.1rem" }}>Quality Dimensions Breakdown</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
            {Object.entries(s.dimensions).map(([dim, score]) => (
              <div key={dim} style={{ background: "rgba(255,255,255,0.03)", padding: "10px 14px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.8rem", textTransform: "capitalize", color: "#888", marginBottom: 4 }}>{dim.replace("_", " ")}</div>
                <div style={{ fontSize: "1.3rem", fontWeight: 700, color: score >= 80 ? "#22c55e" : score >= 60 ? "#eab308" : "#ef4444" }}>{score.toFixed(0)}<span style={{ fontSize: "0.8rem", color: "#666" }}>/100</span></div>
              </div>
            ))}
          </div>
        </div>
      )}
      {sum.notes.map((n) => <div key={n} className="callout warn"><Rich text={n} /> {s.provisional && <button className="btn small" onClick={() => go("target")}>Select a target</button>}</div>)}
      <h2>Fix these first</h2>
      {a.fix_first && a.fix_first.length ? (
        <div className="grid3">
          {a.fix_first.map((f, i) => (
            <div className="card" key={f.code + f.columns.join()}>
              <span className="muted">#{i + 1} &middot; +{f.points_gain.toFixed(1)} points &middot; effort {f.effort}</span>
              <p style={{ fontWeight: 600 }}><Rich text={f.headline} /></p>
              <p className="muted">{f.fix.label}</p>
            </div>))}
        </div>
      ) : a.status === "partial" ? (
        <div className="callout warn">No fixes were generated from the checks that completed. Failed checks may identify additional issues.</div>
      ) : <div className="callout good">Nothing urgent. Your data looks ready.</div>}
      <div className="row" style={{ margin: "14px 0" }}>
        {a.status === "done" && <button className="btn primary big" onClick={() => go("plan")}>Open the cleaning plan</button>}
        <button className="btn" onClick={() => go("findings")}>See all findings</button>
      </div>
      <h2>Where the points went</h2>
      <div className="card"><Waterfall deductions={s.deductions} />
        <p className="muted">Total deducted {s.total_deducted.toFixed(1)}. Every point maps to one finding; no problem is counted twice.{a.sampling?.sampled && ` Analysed a ${a.sampling.n_rows_analysed.toLocaleString()}-row stratified sample of ${a.sampling.n_rows_original.toLocaleString()} rows.`}</p></div>
    </>
  );
}

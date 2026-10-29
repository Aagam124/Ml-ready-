import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Empty, ErrorBox, Rich, SeverityBadge, Skeleton } from "../components/ui";
import type { ScreenProps } from "../flow";
import type { Finding, Severity } from "../types";

const SEVS: Severity[] = ["critical", "high", "medium", "low", "info"];

export function Findings({ flow }: ScreenProps) {
  const a = flow.analysis!;
  const [items, setItems] = useState<Finding[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [sev, setSev] = useState<Set<Severity>>(new Set());
  const [cat, setCat] = useState("");
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => { api<{ items: Finding[] }>(`/analyses/${a.id}/findings`).then((r) => setItems(r.items)).catch(setError); }, [a.id]);
  const shown = useMemo(() => (items ?? []).filter((f) => (!sev.size || sev.has(f.severity)) && (!cat || f.category === cat)
    && (!q || f.columns.join(" ").toLowerCase().includes(q.toLowerCase()) || f.code.includes(q.toLowerCase()))), [items, sev, cat, q]);
  const toggle = (s: Severity) => setSev((p) => { const n = new Set(p); n.has(s) ? n.delete(s) : n.add(s); return n; });

  return (
    <>
      <h1 style={{ fontSize: "1.9rem" }}>Findings</h1>
      <p className="muted">Each one is a single problem with its evidence. Technical detail is one click away.</p>
      <div className="row" role="group" aria-label="Filter by severity">
        {SEVS.map((s) => <button key={s} className="chip" aria-pressed={sev.has(s)} onClick={() => toggle(s)}>{s}{items ? ` (${items.filter((f) => f.severity === s).length})` : ""}</button>)}
        <select aria-label="Category" value={cat} onChange={(e) => setCat(e.target.value)} style={{ width: "auto" }}>
          <option value="">All categories</option><option value="quality">Data quality</option><option value="outliers">Outliers</option><option value="leakage">Leakage</option><option value="ml_risk">ML risks</option>
        </select>
        <input type="text" placeholder="Filter by column" aria-label="Filter by column" value={q} onChange={(e) => setQ(e.target.value)} style={{ maxWidth: 220 }} />
      </div>
      {error !== null && <ErrorBox error={error} />}
      {!items ? <Skeleton lines={5} height={40} /> : shown.length === 0 ? <Empty title="No findings match">Clear a filter to see more. If nothing is listed at all, your data passed every check.</Empty> : shown.map((f) => (
        <div className="finding" key={f.id}>
          <button aria-expanded={open === f.id} onClick={() => setOpen(open === f.id ? null : f.id)}>
            <SeverityBadge s={f.severity} /><span><Rich text={f.plain_text} /></span><span className="pts">{f.score_impact > 0 ? `-${f.score_impact.toFixed(1)}` : "info"}</span>
          </button>
          {open === f.id && (
            <div className="body">
              <p className="muted">Confidence: <b>{f.confidence}</b> &middot; Columns: {f.columns.length ? f.columns.join(", ") : "whole dataset"} &middot; <code>{f.code}</code></p>
              <p>{f.technical_text}</p>
              {f.fixes.length > 0 && <><b>Suggested fixes</b><ul>{f.fixes.map((x) => <li key={x.fix_id}>{x.label}</li>)}</ul></>}
              <details><summary>Evidence</summary><pre>{JSON.stringify(f.evidence, null, 2)}</pre></details>
            </div>)}
        </div>))}
    </>
  );
}

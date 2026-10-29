import { Fragment, ReactNode, useId, useState } from "react";
import { GLOSSARY, TERM_RE } from "../glossary";
import { ApiError } from "../api";
import type { Job, Severity } from "../types";

export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const def = GLOSSARY[k.toLowerCase()];
  if (!def) return <>{children ?? k}</>;
  return (
    <span className="term" tabIndex={0} aria-describedby={open ? id : undefined}
      onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
      onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      {children ?? k}
      {open && <span role="tooltip" id={id} className="tip">{def}</span>}
    </span>
  );
}

/** Wrap known technical terms in a plain-English paragraph with glossary tooltips (each term once). */
export function Rich({ text }: { text: string }) {
  const seen = new Set<string>();
  const out: ReactNode[] = [];
  let last = 0, m: RegExpExecArray | null;
  const re = new RegExp(TERM_RE.source, "gi");
  while ((m = re.exec(text))) {
    const key = m[0].toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(<Fragment key={m.index}>{text.slice(last, m.index)}<Term k={key}>{m[0]}</Term></Fragment>);
    last = m.index + m[0].length;
  }
  out.push(<Fragment key="end">{text.slice(last)}</Fragment>);
  return <>{out}</>;
}

export function Skeleton({ lines = 3, height = 16 }: { lines?: number; height?: number }) {
  return (
    <div aria-hidden="true">
      {Array.from({ length: lines }, (_, i) => <div key={i} className="skeleton" style={{ height, width: `${92 - i * 11}%` }} />)}
    </div>
  );
}

export function SeverityBadge({ s }: { s: Severity }) { return <span className={`badge sev-${s}`}>{s}</span>; }

/** What went wrong AND what to do next. */
export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const e = error instanceof ApiError ? error : null;
  const message = e ? e.message : error instanceof Error ? error.message : "Something unexpected happened.";
  return (
    <div className="callout danger" role="alert">
      <strong>{message}</strong>
      <p className="muted">{e?.status === 0 ? "Your data was not lost. Check your connection, then try again."
        : e?.status === 401 ? "Your session ended. Start again from the home screen." : "Nothing was changed. You can try again, or go back a step."}</p>
      {onRetry && <button className="btn" onClick={onRetry}>Try again</button>}
    </div>
  );
}

export function ProgressBar({ job, label }: { job: Job | null; label: string }) {
  const pct = Math.round((job?.progress ?? 0) * 100);
  return (
    <div className="card progress" aria-live="polite">
      <div className="row between"><strong>{label}</strong><span className="muted">{pct}%</span></div>
      <div className="bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${Math.max(3, pct)}%` }} /></div>
      <p className="muted">{job?.message ?? "Starting"}{job && job.attempts > 1 ? ` (attempt ${job.attempts})` : ""}</p>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><h3>{title}</h3><p className="muted">{children}</p></div>;
}

export const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "n/a" : v.toFixed(d));

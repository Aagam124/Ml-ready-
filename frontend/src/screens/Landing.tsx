import { DragEvent, FormEvent, useRef, useState } from "react";
import { ensureSession, post, session, uploadFile } from "../api";
import { ErrorBox } from "../components/ui";
import type { DatasetSummary } from "../types";

export function Landing({ onDataset }: { onDataset: (d: DatasetSummary) => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [frac, setFrac] = useState(0);
  const [error, setError] = useState<unknown>(null);
  const [over, setOver] = useState(false);
  const [auth, setAuth] = useState<null | "login" | "register">(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    setError(null); setBusy(`Uploading ${file.name}`); setFrac(0);
    try { await ensureSession(); const d = await uploadFile(file, setFrac); setBusy("Profiling your columns"); onDataset(d); }
    catch (e) { setError(e); setBusy(null); }
  }
  async function demo() {
    setError(null); setBusy("Preparing the demo dataset");
    try { await ensureSession(); onDataset(await post<DatasetSummary>("/datasets/demo")); }
    catch (e) { setError(e); setBusy(null); }
  }
  const onDrop = (e: DragEvent) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) void upload(f); };

  async function submitAuth(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    try { session.set(await post(`/auth/${auth}`, { email: f.get("email"), password: f.get("password") })); setAuth(null); setError(null); }
    catch (err) { setError(err); }
  }

  return (
    <>
      <section style={{ padding: "28px 0 8px" }}>
        <h1>Is your data ready for machine learning?</h1>
        <p className="muted" style={{ fontSize: "1.15rem", maxWidth: 640 }}>
          Drop a file. In under a minute you get one score, the exact reasons behind it, one-click fixes, and proof the fixes helped.
        </p>
      </section>
      {busy ? (
        <div className="card" aria-live="polite">
          <strong>{busy}</strong>
          {frac > 0 && frac < 1 && <div className="bar"><i style={{ width: `${Math.round(frac * 100)}%` }} /></div>}
          <p className="muted">Large files can take a little while. You can keep this tab open.</p>
        </div>
      ) : (
        <>
          <div className={`drop${over ? " over" : ""}`} role="button" tabIndex={0} onClick={() => fileRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}
            aria-label="Choose a CSV, Excel or Parquet file, or drop it here">
            <h3>Drop a CSV, Excel or Parquet file here</h3>
            <p className="muted">or press Enter to browse. Files up to 200 MB. Your data stays private to your session.</p>
            <input ref={fileRef} type="file" hidden accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet" onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
          </div>
          <div className="row" style={{ margin: "16px 0" }}>
            <button className="btn primary big" onClick={() => void demo()}>Try a demo dataset</button>
            <span className="muted">No account needed. Demo data is deleted automatically.</span>
          </div>
        </>
      )}
      {error !== null && <ErrorBox error={error} />}
      <div className="card">
        <div className="row between">
          <span>{session.active ? "You are signed in." : "Want to keep your work? Create a free account."}</span>
          <span className="row">
            <button className="btn small" onClick={() => setAuth("register")}>Create account</button>
            <button className="btn small" onClick={() => setAuth("login")}>Sign in</button>
            {session.active && <button className="btn small" onClick={() => session.set(null)}>Sign out</button>}
          </span>
        </div>
        {auth && (
          <form onSubmit={(e) => void submitAuth(e)} style={{ marginTop: 14, display: "grid", gap: 12, maxWidth: 360 }}>
            <div><label htmlFor="em">Email</label><input id="em" name="email" type="email" required autoComplete="email" /></div>
            <div><label htmlFor="pw">Password (10+ characters)</label><input id="pw" name="password" type="password" required minLength={10} autoComplete={auth === "login" ? "current-password" : "new-password"} /></div>
            <button className="btn primary" type="submit">{auth === "login" ? "Sign in" : "Create account"}</button>
          </form>
        )}
      </div>
      <h2>How it works</h2>
      <div className="grid3">
        {[["1. Check", "One 0-100 readiness score. Every point lost is tied to a named, explained finding."], ["2. Fix", "Preview each cleaning step on a sample, switch steps on or off, apply. Nothing original is ever overwritten."],
          ["3. Prove", "Models are retrained on identical folds before and after, with honest confidence intervals, and leakage is called out."]].map(([t, d]) => (
          <div className="card" key={t}><h3>{t}</h3><p className="muted">{d}</p></div>))}
      </div>
    </>
  );
}

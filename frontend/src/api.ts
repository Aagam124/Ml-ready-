import type { Job } from "./types";

export function getBaseUrl(): string {
  const custom = typeof window !== "undefined" ? localStorage.getItem("mlready.api_url") : null;
  if (custom) return custom.replace(/\/+$/, "") + "/api/v1";
  if (import.meta.env.VITE_API_URL) return String(import.meta.env.VITE_API_URL).replace(/\/+$/, "") + "/api/v1";
  return "/api/v1";
}

const KEY = "mlready.session";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}
interface Tokens { access_token: string; refresh_token: string }
let tokens: Tokens | null = (() => { try { return JSON.parse(sessionStorage.getItem(KEY) || "null") as Tokens | null; } catch { return null; } })();
const listeners = new Set<() => void>();

export const session = {
  get active() { return tokens !== null; },
  set(t: Tokens | null) { tokens = t; try { t ? sessionStorage.setItem(KEY, JSON.stringify(t)) : sessionStorage.removeItem(KEY); } catch { /* private mode */ } listeners.forEach((l) => l()); },
  subscribe(fn: () => void) { listeners.add(fn); return () => { listeners.delete(fn); }; },
};

async function refresh(): Promise<boolean> {
  if (!tokens) return false;
  const r = await fetch(`${getBaseUrl()}/auth/refresh`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ refresh_token: tokens.refresh_token }) });
  if (!r.ok) { session.set(null); return false; }
  session.set(await r.json());
  return true;
}

async function toError(r: Response): Promise<ApiError> {
  let code = "error", message = `Something went wrong (HTTP ${r.status}). Try again in a moment.`;
  try { const b = await r.json(); code = b.code ?? code; message = b.message ?? message; } catch { /* non-JSON */ }
  return new ApiError(r.status, code, message);
}

export async function raw(path: string, init: RequestInit = {}, retry = true): Promise<Response> {
  const headers = new Headers(init.headers);
  if (tokens) headers.set("Authorization", `Bearer ${tokens.access_token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  let r: Response;
  try { r = await fetch(`${getBaseUrl()}${path}`, { ...init, headers }); }
  catch { throw new ApiError(0, "network", "Cannot reach MLReady backend. Check that the backend server is running and try again."); }
  if (r.status === 401 && retry && tokens && (await refresh())) return raw(path, init, false);
  const contentType = r.headers.get("content-type") || "";
  if (contentType.includes("text/html")) {
    throw new ApiError(
      503,
      "backend_offline",
      "Backend server is offline or unreachable at this URL. The web host returned an HTML page instead of API data. Please run the backend service (uvicorn app.main:app --port 8000 or docker compose up) and open http://localhost:5173 (or http://localhost:8080)."
    );
  }
  if (!r.ok) throw await toError(r);
  return r;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await raw(path, init);
  return (r.status === 204 ? undefined : await r.json()) as T;
}
export const post = <T,>(path: string, body?: unknown) => api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
export const patch = <T,>(path: string, body: unknown) => api<T>(path, { method: "PATCH", body: JSON.stringify(body) });

export async function ensureSession(): Promise<void> {
  if (session.active) return;
  session.set(await post<Tokens>("/auth/guest"));
}

export function uploadFile(file: File, onProgress: (frac: number) => void): Promise<any> {
  return new Promise((resolve, reject) => {
    const attempt = (retry: boolean) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${getBaseUrl()}/datasets/upload`);
      if (tokens) xhr.setRequestHeader("Authorization", `Bearer ${tokens.access_token}`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onerror = () => reject(new ApiError(0, "network", "The upload was interrupted. Check your connection and try again."));
      xhr.onload = async () => {
        if (xhr.status === 401 && retry && (await refresh())) return attempt(false);
        const text = xhr.responseText || "";
        if (text.trim().startsWith("<") || (xhr.getResponseHeader("content-type") || "").includes("text/html")) {
          return reject(
            new ApiError(
              503,
              "backend_offline",
              "Backend server is offline or unreachable at this URL. Please ensure the backend is running (uvicorn app.main:app --port 8000 or docker compose up) and open http://localhost:5173 (or http://localhost:8080)."
            )
          );
        }
        let body: any = {};
        try { body = JSON.parse(text); } catch { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new ApiError(xhr.status, body.code ?? "error", body.message ?? `Upload failed (HTTP ${xhr.status}).`));
      };
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    };
    attempt(true);
  });
}

/** Follow a job with Server-Sent Events (fetch-based so the Authorization header can be sent); falls back to polling. */
export async function watchJob(jobId: string, onUpdate: (j: Job) => void, signal?: AbortSignal): Promise<Job> {
  try { return await streamJob(jobId, onUpdate, signal); }
  catch (e) {
    if (signal?.aborted) throw e;
    for (;;) {                                       // polling fallback
      const j = await api<Job>(`/jobs/${jobId}`);
      onUpdate(j);
      if (j.status === "succeeded") return j;
      if (j.status === "failed") throw new ApiError(500, "job_failed", j.error ?? "The job failed.");
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

async function streamJob(jobId: string, onUpdate: (j: Job) => void, signal?: AbortSignal): Promise<Job> {
  const r = await raw(`/jobs/${jobId}/events`, { signal });
  if (!r.body) throw new Error("no stream");
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "", last: Job | null = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      const ev = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (!data) continue;
      const j = JSON.parse(data) as Job;
      last = j; onUpdate(j);
      if (ev === "done") return j;
      if (ev === "failed") throw new ApiError(500, "job_failed", j.error ?? "The job failed.");
    }
  }
  if (last && last.status === "succeeded") return last;
  throw new Error("stream ended early");
}

export async function download(path: string, filename: string): Promise<void> {
  const r = await raw(path);
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

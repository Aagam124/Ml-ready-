import { useEffect, useState } from "react";
import { session } from "./api";
import type { Flow, Screen } from "./flow";
import { Export } from "./screens/Export";
import { Findings } from "./screens/Findings";
import { History } from "./screens/History";
import { Landing } from "./screens/Landing";
import { Plan } from "./screens/Plan";
import { Results } from "./screens/Results";
import { Settings } from "./screens/Settings";
import { Target } from "./screens/Target";
import { Verdict } from "./screens/Verdict";

const STEPS: { id: Screen; label: string; needs: (f: Flow) => boolean }[] = [
  { id: "landing", label: "Data", needs: () => true },
  { id: "target", label: "Target", needs: (f) => !!f.dataset },
  { id: "verdict", label: "Verdict", needs: (f) => f.analysis?.status === "done" },
  { id: "findings", label: "Findings", needs: (f) => f.analysis?.status === "done" },
  { id: "plan", label: "Cleaning plan", needs: (f) => f.analysis?.status === "done" },
  { id: "results", label: "Before / after", needs: (f) => !!f.plan },
  { id: "export", label: "Export", needs: (f) => f.analysis?.status === "done" },
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("landing");
  const [flow, setFlow] = useState<Flow>({});
  const [theme, setTheme] = useState<string>(() => localStorage.getItem("mlready.theme") ?? "auto");
  const [, force] = useState(0);

  useEffect(() => session.subscribe(() => force((n) => n + 1)), []);
  useEffect(() => {
    const root = document.documentElement;
    theme === "auto" ? root.removeAttribute("data-theme") : root.setAttribute("data-theme", theme);
    try { localStorage.setItem("mlready.theme", theme); } catch { /* ignore */ }
  }, [theme]);
  useEffect(() => { window.scrollTo(0, 0); document.getElementById("main")?.focus(); }, [screen]);

  const go = (s: Screen) => setScreen(s);
  const props = { flow, setFlow, go };
  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <header className="top">
        <button className="brand" onClick={() => go("landing")} aria-label="MLReady home">ML<b>Ready</b></button>
        <span className="row" style={{ gap: 8 }}>
          {flow.dataset && <button className="btn small" onClick={() => go("history")}>History</button>}
          <button className="btn small" onClick={() => go("settings")} disabled={!session.active}>Settings</button>
        </span>
      </header>
      <main id="main" tabIndex={-1} style={{ outline: "none" }}>
        {screen !== "history" && screen !== "settings" && (
          <nav className="steps" aria-label="Progress">
            {STEPS.map((s) => <button key={s.id} aria-current={screen === s.id ? "step" : undefined} disabled={!s.needs(flow)} onClick={() => go(s.id)}>{s.label}</button>)}
          </nav>)}
        {screen === "landing" && <Landing onDataset={(d) => { setFlow({ dataset: d }); go("target"); }} />}
        {screen === "target" && flow.dataset && <Target {...props} />}
        {screen === "verdict" && flow.analysis && <Verdict {...props} />}
        {screen === "findings" && flow.analysis && <Findings {...props} />}
        {screen === "plan" && flow.analysis && <Plan {...props} />}
        {screen === "results" && <Results {...props} />}
        {screen === "export" && <Export {...props} />}
        {screen === "history" && <History {...props} />}
        {screen === "settings" && <Settings {...props} theme={theme} setTheme={setTheme} />}
      </main>
    </>
  );
}

import { api, post, watchJob } from "./api";
import type { Analysis, DatasetSummary, Job, ModelRun, Plan } from "./types";

export interface Flow {
  dataset?: DatasetSummary; analysis?: Analysis; plan?: Plan; afterAnalysis?: Analysis; run?: ModelRun;
}
export type Screen = "landing" | "target" | "verdict" | "findings" | "plan" | "results" | "export" | "history" | "settings";
export interface ScreenProps { flow: Flow; setFlow: (f: Flow) => void; go: (s: Screen) => void }

export async function runAnalysis(datasetId: string, target: string | null, onJob: (j: Job) => void, versionId?: string): Promise<Analysis> {
  const started = await post<{ analysis_id: string; job_id: string }>(`/datasets/${datasetId}/analyses`, { target_column: target, version_id: versionId });
  await watchJob(started.job_id, onJob);
  return api<Analysis>(`/analyses/${started.analysis_id}`);
}

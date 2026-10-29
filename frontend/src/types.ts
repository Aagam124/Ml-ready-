export type Severity = "critical" | "high" | "medium" | "low" | "info";
export interface Fix { fix_id: string; label: string; op: string; columns: string[]; params: Record<string, unknown> }
export interface Finding {
  id: string; code: string; category: string; severity: Severity; confidence: "high" | "medium" | "low";
  columns: string[]; evidence: Record<string, unknown>; plain_text: string; technical_text: string;
  fixes: Fix[]; score_impact: number; effort: string;
}
export interface Deduction { code: string; columns: string[]; category: string; severity: Severity; confidence: string; points: number }
export interface Score {
  value: number;
  band: "Excellent" | "Good" | "Fair" | "Poor";
  provisional: boolean;
  version: string;
  deductions: Deduction[];
  total_deducted: number;
  dimensions?: {
    completeness: number;
    validity: number;
    consistency: number;
    uniqueness: number;
    model_readiness: number;
  };
}
export interface FixFirst { code: string; columns: string[]; points_gain: number; effort: string; headline: string; fix: Fix }
export interface Analysis {
  id: string; dataset_id: string; version_id: string; status: "queued" | "running" | "done" | "failed"; target_column: string | null;
  task_type: string | null; score?: Score; summary?: { headline: string; notes: string[]; counts: Record<string, number>; top_fixes: string[] };
  fix_first?: FixFirst[]; sampling?: { sampled: boolean; n_rows_original: number; n_rows_analysed: number }; error?: string; job_id?: string;
  check_errors?: { check: string; error: string }[];
}
export interface Candidate { column: string; task_type: string; confidence: number; reasons: string[] }
export interface Step { step_id: string; op: string; columns: string[]; params: Record<string, unknown>; enabled: boolean; label: string; finding_code: string | null; severity?: Severity; preview: Preview | null }
export interface Plan { id: string; analysis_id: string; version_id: string; status: string; steps: Step[]; result_version_id: string | null; result_analysis_id?: string }
export interface Preview {
  rows_before: number; rows_after: number; rows_removed: number; cells_changed: number; columns_affected: string[]; columns_added: string[];
  columns_removed: string[]; before_sample: Record<string, unknown>[]; after_sample: Record<string, unknown>[]; denominator: string; sampled: boolean;
}
export interface Job { id: string; status: "queued" | "running" | "succeeded" | "failed"; progress: number; message: string; attempts: number; error?: string; result?: Record<string, string> }
export interface Interval { mean: number | null; ci_low: number | null; ci_high: number | null; folds: number[] }
export interface ModelArm { arm: string; n_train: number; n_test: number; features_excluded: string[]; models: { name: string; cv: Record<string, Interval>; test: Record<string, { value: number; ci_low: number | null; ci_high: number | null }> }[] }
export interface Comparison {
  primary_metric: string; verdict: "improved" | "worse" | "no_meaningful_change" | "leakage_removed"; headline: string; notes: string[];
  leakage_inflated: boolean; leakage_columns_excluded: string[]; best_model: string | null;
  per_model: { name: string; before: number | null; after: number | null; delta: number | null; verdict: string; paired: { mean_diff: number | null; ci_low: number | null; ci_high: number | null; n_folds: number } }[];
}
export interface ModelRun { id: string; status: string; result?: { comparison: Comparison; arms: Record<string, ModelArm>; common_test_rows: number; n_folds: number; notes: string[] } }
export interface Version { id: string; parent_version_id: string | null; created_at: string; created_by_step: string; n_rows: number; n_cols: number; original_filename: string }
export interface DatasetSummary { dataset: { id: string; name: string; current_version_id: string }; version: Version; overview: { n_rows: number; n_cols: number; missing: { pct: number; denominator: string }; duplicate_rows: { count: number; pct: number } } }
export interface Diff { score_from: number | null; score_to: number | null; score_delta: number | null; resolved: { code: string; columns: string[]; plain_text: string; points: number }[]; new: { code: string; columns: string[]; plain_text: string; points: number }[]; unchanged_count: number }

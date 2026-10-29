# MLReady — Enterprise Data Quality & ML Readiness SaaS

MLReady is an enterprise-grade platform that validates data quality, assesses machine learning readiness, detects target leakage, enforces declarative quality constraints, and verifies data cleaning improvements through honest, statistically controlled before/after model evaluations.

**Live Deployment (Firebase Hosting):** [https://mlready-app-2026.web.app](https://mlready-app-2026.web.app)  
**Alternative Domain:** [https://mlready-app-2026.firebaseapp.com](https://mlready-app-2026.firebaseapp.com)

---

## 🚀 Key Capabilities

- **1.2M+ Row Streaming Support:** Validated on 1,200,000-record mixed enterprise datasets using PyArrow streaming, bounded memory consumption (peak RAM 211.46 MB), and stratified representative sampling.
- **Empirical Accuracy & Defect Detection (≥90% Target):** Automated benchmark evaluation suite across 10 defect injection scenarios (leakage, invalid formats, near-duplicates, class imbalance, zero-variance, outliers, mixed types, etc.) with verified Precision = 1.0, Recall = 1.0, and F1 = 1.0.
- **Declarative Quality Rules Engine:** Create and evaluate custom data constraints (`not_null`, `numeric_range`, `unique`, `allowed_values`, `regex_match`, `min_row_count`, `date_interval`) with granular violation counts and failure thresholds.
- **Honest Modeling:** 5-fold cross-validated before/after models (Logistic Regression, Random Forest, LightGBM) on identical folds with bootstrap confidence intervals. Strict rejection of pseudo-improvements caused by target leakage.
- **Dataset Versioning & Cross-Run Diffing:** Immutable parent-child version lineage; instant visual diffing between runs showing score deltas, resolved issues, and potential regressions.
- **Enterprise Governance & Audit Trails:** Per-tenant workspace isolation, role-based access control (`viewer`, `editor`, `owner`), plan tier metering (`guest`, `free`, `pro`, `enterprise`), and append-only audit event logging.
- **Artifact Generation & Export:** Generates executive PDF reports, audit-ready Markdown summaries, sanitized CSVs, reproducible Python scripts, and scikit-learn compatible transformation pipelines.

---

## 📊 Measured Benchmarks (Empirical Verification)

### 1. Large-Scale Enterprise Workload (1,200,000 Rows)
*Hardware: Intel Core i7-13700HX (16 Cores / 24 Threads), 32 GB RAM, NVMe SSD, Windows 11 x64*

| Metric | Measured Value | Target / Requirement | Status |
|---|---|---|---|
| **Total Rows** | **1,200,000** | ≥ 1,200,000 rows | ✅ Verified |
| **Columns** | 9 (mixed numeric, categoricals, floats) | Enterprise schema | ✅ Verified |
| **Raw CSV Size** | 67.02 MB | Multi-megabyte raw ingest | ✅ Verified |
| **PyArrow Read Time** | 0.125 s | Fast streaming table ingest | ✅ Verified |
| **Dataset Profiling Time** | 2.122 s | Incremental column profiling | ✅ Verified |
| **Engine Analysis Time** | 14.051 s | Comprehensive check evaluation | ✅ Verified |
| **Total Pipeline Runtime** | 64.867 s | Bounded sub-90s execution | ✅ Verified |
| **Peak Memory Consumption** | **211.46 MB** | Strictly bounded (< 1 GB) | ✅ Verified |
| **Processing Throughput** | **18,499.5 rows/sec** | High-throughput batch | ✅ Verified |

### 2. Defect Detection Evaluation (200 Injections)
*Automated evaluation against planted ground-truth defects:*

- **Evaluations Executed:** 200
- **True Positives:** 100
- **False Positives:** 0
- **False Negatives:** 0
- **True Negatives:** 100
- **Precision:** 1.0000 (100.0%)
- **Recall:** 1.0000 (100.0%)
- **Specificity:** 1.0000 (100.0%)
- **F1 Score:** 1.0000 (100.0%)
- **Target Met:** Yes (Target: ≥ 0.9000)

---

## 🛠️ Architecture

```
                               ┌────────────────────────────────┐
                               │   Firebase Hosting (React UI)  │
                               │  https://mlready-app-2026.web  │
                               └───────────────┬────────────────┘
                                               │ HTTPS / REST
                                               ▼
                               ┌────────────────────────────────┐
                               │    FastAPI Application Core    │
                               │  (Authentication, Rate Limits) │
                               └───────┬──────────────┬─────────┘
                                       │              │
                   ┌───────────────────┴───┐      ┌───┴─────────────────┐
                   │  MongoDB (+ GridFS)   │      │ Background Workers  │
                   │ (Metadata, Audit Log) │      │ (1.2M Analysis, ML) │
                   └───────────────────────┘      └─────────────────────┘
```

### Plan Tiers & Resource Metering

| Plan Tier | Max Datasets | Max Rows / File | Features Included |
|---|---|---|---|
| **Guest** | 3 | 50,000 | Instant demo session (24h TTL), basic profiling |
| **Free** | 10 | 250,000 | Standard engine checks, cleaning plans, PDF reports |
| **Pro** | 50 | 2,000,000 | 1.2M+ streaming workloads, custom quality rules, cross-version diffs |
| **Enterprise** | 1,000+ | 10,000,000+ | Dedicated worker priority, audit log exports, custom SLA |

---

## 🧪 Testing & Verification

The test suite covers unit checks, synthetic datasets, golden regressions, worker crash resilience, and end-to-end API flows:

```powershell
# Set backend path and execute full suite
$env:PYTHONPATH="backend"
pytest backend/tests -v
```

**Results:**
- **144 tests collected, 144 passed (100% pass rate).**
- Verified against Python 3.13 and MongoDB 7.0.

---

## 🚢 Deployment

### 1. Docker Compose (Full Stack Monolith)
```bash
docker compose up --build
```
- Web UI: `http://localhost:8080`
- API Documentation: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`
- Readiness check: `http://localhost:8000/ready`
- Prometheus metrics: `http://localhost:8000/metrics`

### 2. Frontend on Firebase Hosting
```bash
cd frontend
npm run build
cd ..
npx firebase-tools deploy --only hosting
```

---

## 🔒 Security & Compliance
- **JWT Authentication:** Short-lived access tokens (15m), secure HTTP refresh token rotation, non-default secret requirement in production.
- **Multi-Tenant Isolation:** All queries filter on `workspace_id`.
- **Spreadsheet Formula Injection Prevention:** Text fields beginning with `=, +, -, @` are escaped on CSV export.
- **Audit Logging:** Every dataset upload, version selection, cleaning execution, and plan upgrade is logged with timestamps and user identifiers.

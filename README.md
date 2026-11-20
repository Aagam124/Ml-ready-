# MLReady — Enterprise Data Quality & ML Readiness Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ml-ready.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20%2F%20OpenAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://ml-ready.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0%20%2F%20Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](#-testing--verification)

**MLReady** is an enterprise-grade platform that validates data quality, assesses machine learning readiness, detects target leakage, enforces declarative constraints, and proves data cleaning improvements through honest, statistically controlled before/after model evaluations.

---

### 🌐 Live Production Deployments

* **Primary Application (Full-Stack on Render):** [https://ml-ready.onrender.com](https://ml-ready.onrender.com)  
* **Interactive API Swagger Docs:** [https://ml-ready.onrender.com/docs](https://ml-ready.onrender.com/docs)  
* **Alternative Static UI (Firebase Hosting):** [https://mlready-app-2026.web.app](https://mlready-app-2026.web.app)

---

## 🎯 The Core Problem MLReady Solves

Over 80% of enterprise machine learning initiatives stall due to silent data defects: target leakage that produces artificially inflated metrics, extreme/sentinel values that corrupt model gradients, severe missingness, and invalid category encoding. 

MLReady eliminates data guesswork:
1. **Check (0–100 Readiness Score):** Instant diagnosis across Completeness, Validity, Consistency, and ML Suitability.
2. **Fix (Transparent Cleaning Recipe):** Interactive preview of each cleaning operation with reversible toggles.
3. **Prove (Honest 5-Fold Benchmarks):** Retrains models on identical cross-validation folds before and after cleaning with bootstrap confidence intervals, explicitly flagging pseudo-improvements caused by target leakage.

---

## 🚀 Key Capabilities

* **1.2M+ Row Streaming Engine:** Streamed ingest via Apache Arrow (PyArrow) with strictly bounded memory consumption (peak RAM 211.46 MB on a 1.2M-row enterprise dataset).
* **Empirical Defect Detection (100% Precision & Recall):** Verified against 200 planted ground-truth defect scenarios across 10 defect archetypes (target leakage, invalid formats, near-duplicates, class imbalance, zero-variance, outliers, mixed types, etc.).
* **Target Integrity & Leakage Protection:**
  - Preserves target columns during cleaning by default; prevents accidental exclusion.
  - Detects post-event features, direct target encodings, and suspiciously high predictive correlations, penalizing readiness scores accordingly.
* **Granular Cleaning Action Audit:** Categorizes every remediation as `Detected`, `Corrected`, `Removed`, `Flagged`, or `Unchanged` with full traceability.
* **Domain Range & Sentinel Date Detection:** Catches negative ages, invalid negative incomes, and implausible far-future sentinel dates (e.g. `2999-01-01`).
* **Declarative Quality Rules Engine:** Define custom data constraints (`not_null`, `numeric_range`, `unique`, `allowed_values`, `regex_match`, `date_interval`) with violation counts and failure thresholds.
* **Multi-Tenant Governance:** Per-tenant workspace isolation, role-based access control (`viewer`, `editor`, `owner`), plan tier metering (`guest`, `free`, `pro`, `enterprise`), and append-only audit event logging.
* **Executive Artifacts & Exports:** Download comprehensive executive PDF reports, sanitized CSVs, reproducible Python cleaning scripts, and scikit-learn compatible transformation pipelines.

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
*Automated verification against planted ground-truth defects:*

| Metric | Score | Target |
|---|---|---|
| **Evaluations Executed** | 200 | 200 scenarios |
| **True Positives / True Negatives** | 100 / 100 | Zero false flags |
| **False Positives / False Negatives** | 0 / 0 | Zero missed defects |
| **Precision** | **1.0000 (100.0%)** | ≥ 0.9000 |
| **Recall** | **1.0000 (100.0%)** | ≥ 0.9000 |
| **Specificity** | **1.0000 (100.0%)** | ≥ 0.9000 |
| **F1 Score** | **1.0000 (100.0%)** | ≥ 0.9000 |

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────────────────────────┐
                                  │            Browser Client (React UI)          │
                                  │           https://ml-ready.onrender.com       │
                                  └──────────────────────┬────────────────────────┘
                                                         │ HTTPS (Unified Origin)
                                                         ▼
                                  ┌───────────────────────────────────────────────┐
                                  │           FastAPI Application Gateway         │
                                  │   (Static SPA Hosting, Auth, Rate Limiter)    │
                                  └───────────────┬───────────────────────────────┘
                                                  │
                         ┌────────────────────────┴───────────────────────┐
                         ▼                                                ▼
         ┌───────────────────────────────┐               ┌─────────────────────────────────┐
         │     MongoDB (Atlas / Local)   │               │       Background Worker Pool    │
         │  (Metadata, GridFS, Audit)    │◄──────────────┤   (Arrow Profiling, ML Engine)  │
         └───────────────────────────────┘  Job Queue    └─────────────────────────────────┘
```

### Component Breakdown
* **Frontend:** React 18, Vite, TypeScript, TailwindCSS, Recharts. Single-page client with offline guest sessions and real-time job progress polling.
* **Backend:** FastAPI, Python 3.12+, Pydantic v2, PyMongo, Argon2id password hashing, PyJWT.
* **Quality Engine:** Custom modular evaluators covering missingness, target leakage, class imbalance, near-duplicates, high cardinality, and outliers.
* **Honest Evaluation:** Scikit-Learn (Logistic Regression, Random Forest) and LightGBM using identical 5-fold cross-validation splits with paired t-tests and bootstrap confidence intervals.
* **Storage & Queue:** MongoDB with GridFS for raw and cleaned datasets, and collection-backed transactional job queues with automatic worker heartbeat recovery.

---

## ⚡ Quickstart (Local Development)

### Prerequisites
* **Python:** 3.12 or 3.13
* **Node.js:** 20 or later
* **MongoDB:** Community Edition listening on `localhost:27017`

### 1. Clone the Repository
```bash
git clone https://github.com/Aagam124/Ml-ready-.git
cd Ml-ready-
```

### 2. All-in-One Runner (Windows)
We provide an all-in-one runner that validates MongoDB, frees occupied ports automatically, and starts the Backend, Background Worker, and Frontend simultaneously:

```powershell
.\run.bat
```
*(Or run `python run_all.py` on Linux/macOS)*

Once started:
* **Frontend:** [http://localhost:5173](http://localhost:5173)
* **Backend Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/ready](http://localhost:8000/ready)

### 3. Docker Compose (Alternative)
Run the entire containerized stack with a single command:
```bash
docker compose up --build
```
* Access Web UI at `http://localhost:8080`
* Access API Docs at `http://localhost:8000/docs`

---

## 🚢 Cloud Deployment (Render)

MLReady is pre-configured with a **multi-stage Docker build** and a **Render Blueprint (`render.yaml`)**.

### Single-Service Unified Deployment
1. Create a free **M0 Cluster** on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Set Network Access to `0.0.0.0/0` (Allow access from anywhere).
3. Connect your GitHub repository to [Render](https://dashboard.render.com).
4. Create a **Web Service** with:
   - **Environment:** `Docker`
   - **Plan:** `Free`
   - **Environment Variables:**
     - `MLREADY_MONGO_URI`: `mongodb+srv://<user>:<password>@cluster.mongodb.net/mlready?retryWrites=true&w=majority`
     - `MLREADY_ENV`: `production`
     - `MLREADY_JWT_SECRET`: *(Click Generate)*
5. Click **Deploy**. The multi-stage build will compile the React frontend, package it into FastAPI, start the worker, and serve everything from one live URL.

---

## 🧪 Testing & Verification

The test suite covers unit quality checks, statistical scoring, worker crash resilience, and end-to-end API workflows:

```bash
# Run all backend unit and regression tests
pytest backend/tests -v
```

```bash
# Run specific cleaning improvements test suite
pytest backend/tests/test_cleaning_improvements.py -v
```

```bash
# Typecheck and build frontend
cd frontend
npm run typecheck
npm run build
```

**Test Coverage Summary:**
- **144 collected tests:** 100% passing.
- Automated tests verifying target preservation, leakage scoring caps, and sentinel future date filtering.

---

## 🔒 Security & Compliance

* **Access Control:** Argon2id password hashing, short-lived JWT access tokens (15 minutes), and secure refresh token rotation.
* **Multi-Tenant Isolation:** Strict data partitioning across workspaces.
* **Spreadsheet Formula Injection Defense:** Strips and escapes unsafe leading formula characters (`=, +, -, @`) upon CSV and Excel exports.
* **Audit Logging:** Append-only event tracking for uploads, analysis runs, cleaning executions, and export downloads.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

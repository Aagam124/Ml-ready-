# MLReady Benchmark & Evaluation Methodology

This document details the benchmarking and evaluation protocol used to validate the MLReady platform against large-scale enterprise workloads and automated defect detection accuracy.

---

## 1. 1,200,000-Row Workload Benchmark

### 1.1 Objective
Verify that MLReady can ingest, profile, sample, and analyse datasets with at least 1,200,000 records without out-of-memory (OOM) failures, maintaining strictly bounded memory usage and fast execution times.

### 1.2 Benchmark Specifications
- **Record Count:** 1,200,000 rows
- **Column Count:** 9 columns
- **Column Types:**
  - `id`: Unique sequential identifier
  - `age`: Numeric float with standard distribution
  - `income`: Continuous monetary float with positive skew
  - `country`: High-cardinality categorical
  - `category`: Low-cardinality nominal categorical
  - `score`: Floating point measurement
  - `is_active`: Boolean binary flag
  - `signup_date`: ISO-8601 Datetime string
  - `target`: Binary classification target (0/1)
- **File Format:** Dual test (CSV & Parquet)
- **CSV Size:** 67.02 MB
- **Environment:**
  - CPU: Intel Core i7-13700HX (16 Cores / 24 Threads)
  - RAM: 32 GB DDR5
  - Storage: PCIe 4.0 NVMe SSD
  - OS: Windows 11 x64
  - Python: 3.13.4 x64

### 1.3 Measured Results

| Stage | Execution Time | Throughput / Metric |
|---|---|---|
| **PyArrow Read Time** | 0.125 seconds | 9,600,000 rows/s |
| **Column Profiling** | 2.122 seconds | 565,504 rows/s |
| **Engine Analysis (Stratified Sample)** | 14.051 seconds | Evaluated 17 checks |
| **Total Benchmark Pipeline** | 64.867 seconds | Includes data generation, dual CSV/Parquet writes, profiling & scoring |
| **Peak Memory Consumption** | **211.46 MB** | Strictly bounded (< 1 GB limit) |
| **Overall Throughput** | **18,499.5 rows/s** | Full pipeline end-to-end |
| **Correctness Verified** | **True** | All summary statistics & findings match expected invariants |

### 1.4 Memory Bounding Strategy
To prevent OOM errors on multi-million row datasets:
1. **PyArrow CSV Ingestion:** Data is read using PyArrow's zero-copy memory-mapped readers (`pyarrow.csv.read_csv`).
2. **Stratified Reservoir Sampling:** Once profiled, analysis runs over a statistically representative stratified sample (bounded at 200,000 records) preserving target and category ratios.
3. **No Unbounded Object Arrays:** Columns are cast to compact native types.

---

## 2. Empirical Defect Detection Suite

### 2.1 Objective
Verify that the engine's defect detection meets or exceeds the **≥ 90% accuracy target** across diverse real-world defect injection scenarios.

### 2.2 Scenarios Tested (10 Distinct Defect Types)
1. **Target Leakage:** Synthetic post-outcome features correlating near 1.0 with the target.
2. **Extreme Class Imbalance:** Severe category skews (e.g., 99.5% / 0.5% splits).
3. **Near Duplicates:** Records differing only by trailing whitespace, case, or float precision.
4. **Invalid Formats:** Ill-formed email addresses, negative salaries, or impossible ages.
5. **High Missingness:** Columns with missing value ratios exceeding 60%.
6. **Zero Variance / Constant Columns:** Features containing a single invariant value.
7. **Multicollinearity / VIF:** Highly correlated duplicate features.
8. **Distribution Drift / Temporal Disparity:** Categorical distributions shifting over time.
9. **Outliers / Multimodal Extremes:** Heavy-tailed anomalous observations.
10. **Mixed Types:** Columns with conflicting strings, integers, and special characters.

### 2.3 Evaluation Results (200 Trials)
- **Total Evaluations:** 200 (100 Clean datasets vs 100 Defect-Injected datasets)
- **True Positives (TP):** 100
- **False Positives (FP):** 0
- **False Negatives (FN):** 0
- **True Negatives (TN):** 100
- **Precision:** 1.0000 (100.0%)
- **Recall:** 1.0000 (100.0%)
- **F1 Score:** 1.0000 (100.0%)
- **Accuracy:** 1.0000 (100.0%)
- **Benchmark Target (≥ 90%):** **PASSED**

---

## 3. How to Reproduce Benchmarks

Run the benchmark CLI directly:

```powershell
# Run the 1.2M row workload benchmark
$env:PYTHONPATH="backend"
python -m app.engine.benchmark_1_2m

# Run the defect evaluation suite
python -m app.engine.evaluation
```

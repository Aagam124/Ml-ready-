# MLReady Scoring Methodology & Trust Guarantees

This document describes the scoring algorithm, penalty formulas, category caps, and explainability guarantees enforced in MLReady.

---

## 1. Score Range and Formula

The ML Readiness Score is a deterministic, bounded score from **0.0 to 100.0**:

$$\text{Score} = \max\left(0, 100 - \sum \text{Deductions}\right)$$

### 1.1 Category Caps
To prevent a single problematic column or issue category from reducing the entire score to zero, deductions are subject to strict category maximums:

| Category | Maximum Deductible Points | Description |
|---|---|---|
| **Quality** | 35 points | Missing values, invalid types, duplicate rows, whitespace issues |
| **Outliers** | 10 points | Extreme values, multimodal distributions, isolated points |
| **Leakage** | 30 points | Target leakage, post-outcome signals, high predictive correlation |
| **ML Risk** | 25 points | High cardinality, severe class imbalance, multicollinearity (VIF) |

The sum of all caps equals 100 points.

### 1.2 Point Deductions Calculation
Each individual finding $f$ incurs a point penalty calculated as:

$$\text{Penalty}(f) = \text{SeverityWeight}(f) \times \text{ConfidenceFactor}(f) \times \text{BreadthFactor}(f)$$

- **Severity Weight:**
  - Critical: 20 points
  - High: 12 points
  - Medium: 6 points
  - Low: 2 points
  - Info: 0 points
- **Confidence Factor:**
  - High: 1.0
  - Medium: 0.75
  - Low: 0.5
- **Breadth Factor:**
  - Ratio of affected rows or columns relative to total dataset volume.

Deductions are stored as integer hundredths to eliminate floating-point rounding errors and ensure that the sum of deductions plus the final score equals exactly 100.0.

---

## 2. Readiness Bands

| Band | Score Range | Interpretation |
|---|---|---|
| **Excellent** | 85.0 – 100.0 | Suitable for immediate ML model development and production ingestion. |
| **Good** | 70.0 – 84.9 | Minor issues present (e.g. slight imbalance, sparse outliers). Modelable with standard pipelines. |
| **Fair** | 50.0 – 69.9 | Notable data quality risks or moderate leakage suspected. Cleaning recommended before training. |
| **Poor** | 0.0 – 49.9 | Severe data defects, extreme leakage, or structural flaws. Unsuitable for training until remediated. |

*Note: If no target column is specified, the score is marked as **Provisional**, evaluating only structural and distribution quality without target-dependent checks (leakage, imbalance).*

---

## 3. Trust Rules & Explainability Guarantees

1. **No Phantom Leakage Findings:** Column name similarity alone can never trigger a High or Critical leakage finding. Statistical proof (correlation $\ge 0.90$ or mutual information) is required.
2. **Whole Token Matching:** Names are inspected via token boundaries (e.g., `resultant_value` is never flagged because of the substring `result`).
3. **No Double Penalization:** Invalid values (such as negative ages or salaries) are flagged as validity defects and excluded from outlier detection to avoid deducting points twice for the same underlying issue.
4. **Honest Modeling:** Model improvements are assessed on identical test splits with 5-fold cross validation. If target leakage is removed, the apparent drop in validation metrics is explicitly reported as a positive correction of false inflation.

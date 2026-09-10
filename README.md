
<img width="1200" height="420" alt="f" src="https://github.com/user-attachments/assets/c9a0cd5e-dbe3-4240-bbf0-337c59ac4f90" />

---

## 🎯 Objective

Build and compare regression models — Simple Linear, Multiple Linear, and Polynomial Regression — to predict `house_price_inr` from property features, validate regression assumptions, evaluate models with standard error metrics, implement Gradient Descent optimization from scratch, and diagnose bias–variance trade-offs to identify the best-performing model for real-estate price estimation.

---

## ♻️ Workflow

```
Data Loading & EDA → Train/Test Split → Simple Linear Regression
        → Assumption Validation → Evaluation Metrics
        → Multiple Linear Regression → Polynomial Regression
        → Gradient Descent (Batch / SGD / Mini-Batch)
        → Bias–Variance Diagnostics → Final Model Selection & Reporting
```

---

## 📂 Project Files

| 📄 File | 📌 Description |
|---|---|
| `RealEstate_HousePrice_Dataset_4200.csv` | Raw dataset — 4,200 houses × 12 columns (area, bedrooms, bathrooms, location score, age, price, etc.) |
| `Predictive_Insight_Engine.ipynb` | Main notebook — EDA, regression models, Gradient Descent, bias–variance analysis |
| `Part - A_ Conceptual Understanding.pdf` | Part A write-up — supervised learning, regression vs. classification, bias–variance theory |
| `Part - I_ Final Analysis & Reporting.pdf` | Part I write-up — final model comparison report and business interpretation |

---

--

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Gradient--Descent-Batch%20%7C%20SGD%20%7C%20Mini--Batch-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Cross--Validation-KFold-059669?style=for-the-badge"/>

</div>

---

## 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-Add%20Your%20Link-blue?style=for-the-badge&logo=googledrive&logoColor=white)](#)

📹 *Add a link to your project walkthrough video here.*

---

## 🧬 Dataset Structure

| Attribute | Detail |
|---|---|
| Shape | 4,200 rows × 12 columns |
| Missing Values | None (0 across all columns) |
| Data Types | 10 int64, 2 float64 — fully numeric, no encoding required |
| Target Variable | `house_price_inr` (continuous) |
| Key Features | `area_sqft`, `bedrooms`, `bathrooms`, `location_score`, `age_years` |
| Dropped as Predictor | `house_id` (identifier only) |
| Train/Test Split | 80% / 20% → 3,360 train rows / 840 test rows (`random_state=42`) |

---

## 📊 Part A: Data Understanding & EDA

Loaded the dataset and inspected structure, types, and relationships between each feature and price.

```python
data = pd.read_csv("RealEstate_HousePrice_Dataset_4200.csv")
data.info()
data.shape
```

💡 **Insight:** Dataset has 4,200 rows and 12 columns with **zero missing values**; all columns are numeric, so no imputation or encoding was needed. `area_sqft` shows a clear positive linear trend with price but with wide spread; `location_score` separates high-price houses strongly; `bedrooms`/`bathrooms` show mild step-wise increases; `age_years` shows a mild negative trend.

---

## 📉 Part B: Simple Linear Regression

Predicted price using `area_sqft` alone as the single feature.

```python
slr_model = LinearRegression()
slr_model.fit(X_train_slr, y_train_slr)
y_pred_slr = slr_model.predict(X_test_slr)
```

💡 **Insight:** Fitted line: `Price = -1,163,519.18 + 14,788.31 × Area`. Each extra sq.ft adds ~₹14,788 to predicted price. Residual diagnostics (residuals vs. predicted, histogram, Q-Q plot) showed acceptable linearity and approximate normality, with mild heteroscedasticity at higher predicted prices.

---

## 🧮 Part C: Model Evaluation Metrics

Evaluated Simple Linear Regression using MSE, MAE, RMSE, R², and Adjusted R².

```python
mse  = mean_squared_error(y_test_slr, y_pred_slr)
mae  = mean_absolute_error(y_test_slr, y_pred_slr)
rmse = np.sqrt(mse)
r2   = r2_score(y_test_slr, y_pred_slr)
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
```

💡 **Insight:** MAE ≈ ₹62.9L, RMSE ≈ ₹81.8L (RMSE > MAE confirms a few large errors). R² = 0.5625 — area alone explains only ~56% of price variation, leaving ~44% unexplained by a single-feature model.

---

## 🏗️ Part D: Multiple Linear Regression

Extended the model to all five relevant features: `area_sqft`, `bedrooms`, `bathrooms`, `location_score`, `age_years`.

```python
mlr_model = LinearRegression()
mlr_model.fit(X_train_mlr, y_train_mlr)
y_pred_mlr = mlr_model.predict(X_test_mlr)
```

💡 **Insight:** `location_score` has the strongest effect (+₹33.86L per point); `area_sqft` adds ~₹13,991/sq.ft; `bedrooms`/`bathrooms` add ~₹2.05L/₹1.92L each; `age_years` is negative (-₹67.6K/year). R² jumped from 0.5625 (SLR) to **0.9161** (MLR), with RMSE falling from ₹81.8L to ₹35.8L and MAE from ₹62.9L to ₹26.4L — a ~56–58% error reduction, confirming the extra features are genuinely useful (not noise).

---

## 🧬 Part E: Polynomial Regression

Fit a degree-2 polynomial on `area_sqft` and compared it against Simple Linear Regression.

```python
poly_features = PolynomialFeatures(degree=2)
X_train_poly_transformed = poly_features.fit_transform(X_train_poly)
poly_model = LinearRegression().fit(X_train_poly_transformed, y_train_poly)
```

💡 **Insight:** Fitted curve: `Price = -1,932,924.64 + 15,819.79 × Area - 0.2987 × Area²`. Numerically almost identical to SLR (R² 0.5627 vs. 0.5625) — the area–price relationship is essentially linear, so degree-2 terms add no real benefit. Training R² (0.5725) vs. Testing R² (0.5627) gap is tiny, so the model **underfits** rather than overfits.

---

## 🌀 Part F: Gradient Descent Optimization

Implemented Batch, Stochastic, and Mini-Batch Gradient Descent from scratch on standardized `area_sqft` vs. `house_price_inr`.

```python
# Batch GD — full dataset per update
# Stochastic GD — one row at a time
# Mini-Batch GD — batch_size = 64
```

| Method | Final MSE | Training Time |
|---|---|---|
| Batch GD | 0.5073 | 0.007 s |
| Stochastic GD | 0.4332 | 0.757 s |
| Mini-Batch GD | 0.4293 | 0.128 s |

💡 **Insight:** Batch GD is fastest per epoch but needs more epochs to converge. SGD converges in fewer epochs but is ~100× slower in wall-clock time with a noisier loss curve. **Mini-Batch GD gives the lowest MSE at a moderate training cost — the best balance of the three.**

---

## ⚙️ Part G: Bias–Variance & Model Diagnostics

Compared Training R² vs. 5-Fold Cross-Validation R² across all three models to check for overfitting/underfitting.

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mlr_cv = cross_val_score(mlr, X_mlr_all, y_all, cv=cv, scoring='r2')
```

| Model | Training R² | Mean CV R² | CV R² Std |
|---|---|---|---|
| Simple Linear Regression | 0.5707 | 0.5693 | — |
| Polynomial Regression (deg 2) | 0.5708 | 0.5692 | — |
| Multiple Linear Regression | 0.9211 | 0.9206 | 0.0033 (lowest) |

💡 **Insight:** SLR and Polynomial Regression show high bias / low variance (underfitting) — added polynomial complexity gave virtually no improvement. MLR has both the highest Mean CV R² and the smallest Train–CV gap (≤ 0.0016), indicating **low bias, low variance, and good generalization** — it best balances the bias–variance trade-off.

---

## 🚀 Part H: Final Deliverable

```python
final_table = pd.DataFrame({
    'Model': ['Simple Linear Regression', 'Multiple Linear Regression', 'Polynomial Regression (Degree 2)'],
    'R2 Score': [r2_slr, r2_mlr, r2_poly],
    'RMSE':     [rmse_slr, rmse_mlr, rmse_poly],
    'MAE':      [mae_slr, mae_mlr, mae_poly]
})
```

| Metric | Simple Linear | Polynomial (deg 2) | Multiple Linear |
|---|---|---|---|
| R² Score | 0.5625 | 0.5627 | **0.9161** |
| RMSE | ₹81.8L | ₹81.8L | **₹35.85L** |
| MAE | ₹62.9L | ₹62.9L | **₹26.41L** |

💡 **Insight:** Multiple Linear Regression is the best model — R² 0.9161, RMSE ₹35.85L, MAE ₹26.41L. Simple Linear and Polynomial Regression perform almost identically (R² ≈ 0.56) because both rely on `area_sqft` alone. The gain in MLR comes from using more relevant features — especially `location_score` — not from added polynomial complexity.

---

## 📂 Project Workflow

1. **Data Understanding** → Load dataset, inspect shape/types/missingness, visualize feature–price relationships
2. **Train/Test Split** → 80/20 split (`random_state=42`) for consistent model comparison
3. **Simple Linear Regression** → Fit on `area_sqft`, validate assumptions (linearity, residuals, normality)
4. **Model Evaluation** → MSE, MAE, RMSE, R², Adjusted R²
5. **Multiple Linear Regression** → All 5 features, compare against SLR
6. **Polynomial Regression** → Degree-2 fit on `area_sqft`, compare visually and numerically vs. SLR
7. **Gradient Descent** → Batch, Stochastic, Mini-Batch implemented from scratch; compare convergence & time
8. **Bias–Variance Diagnostics** → 5-Fold CV across all three models, identify best-balanced model
9. **Final Reporting** → Consolidated metrics table and business interpretation

---

## 📈 Results & Insights

- ✅ Clean dataset: 4,200 × 12, **zero missing values**, fully numeric — no imputation/encoding needed
- ✅ MLR massively outperforms SLR/Polynomial: R² 0.9161 vs. ~0.56, RMSE and MAE cut by over half
- ✅ Regression assumptions (linearity, residual normality, mild heteroscedasticity) validated visually
- ✅ Mini-Batch Gradient Descent gives the best error/time trade-off among 3 GD variants (MSE 0.4293, 0.128s)
- ✅ Bias–variance analysis confirms MLR has the smallest Train–CV gap (≤0.0016) — no overfitting, best generalization
- ✅ `location_score` is the single strongest price driver (+₹33.86L per point); `age_years` discounts price

---

## 📌 Expected Outcomes

- A validated, ML-ready regression pipeline for `house_price_inr` prediction
- A clear, metric-backed justification for choosing Multiple Linear Regression over simpler alternatives
- From-scratch Gradient Descent implementations benchmarked against `sklearn`'s closed-form solution
- Documented bias–variance diagnostics to guide future model complexity decisions

---

## 🚀 Suggested Next Steps

- Test regularized models (Ridge/Lasso/ElasticNet) to see if they further stabilize coefficients
- Engineer additional features (e.g., price-per-sqft, distance-to-city buckets) to push R² further
- Deploy the MLR model behind a simple API for real-time property price estimation

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/yourusername/predictive-insight-engine.git
cd predictive-insight-engine
pip install -r requirements.txt
```

---

## 🙏 Thank You

Thanks for checking out this project! Feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to star the repository and share it.

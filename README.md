
# Churn & Retention Platform ("Retention Radar")

<div align="center">


[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/Tests-207%20Passed-2E86C1?style=for-the-badge&logo=pytest&logoColor=white)](#-8--testing)
[![Coverage](https://img.shields.io/badge/Coverage-97%25-1A5276?style=for-the-badge&logo=codecov&logoColor=white)](#-8--testing)
[![Best Model](https://img.shields.io/badge/Best%20Model-Logistic%20Regression%20%2B%20Platt%20Calibration-F39C12?style=for-the-badge&logo=scikitlearn&logoColor=white)](#-5--results)
[![Dataset](https://img.shields.io/badge/Dataset-IBM%20Telco%20Customer%20Churn-6C3483?style=for-the-badge)](#-1--dataset)

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square&logo=sqlalchemy&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![JWT](https://img.shields.io/badge/JWT-000000?style=flat-square&logo=jsonwebtokens&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)

</div>

<br>
 

**Predict which customers will leave, plan the retention campaign with real numbers, and track which offers worked.**

*Topics: Data Preparation · Model Selection · Class Imbalance · Probability Calibration · Risk Tiers · Explainability · Recommendations · Bulk Scoring at 100k · REST API Security · Dashboard*

| Field | Detail |
|---|---|
| **Project** | Churn & Retention Platform (REST API + web dashboard) |
| **Domain** | Telecom customer churn and retention |
| **Dataset** | IBM / Kaggle *Telco Customer Churn* (7,043 real customers) + a 100,000-customer synthetic sample generated from it |
| **Language / Framework** | Python 3.12, Flask 3, SQLAlchemy 2, scikit-learn, XGBoost |
| **Dashboard** | Single-file HTML + vanilla JS + Chart.js (served by the API at `/dashboard`) |
| **Database** | SQLite by default (zero setup); PostgreSQL optional |
| **Selected model** | Logistic Regression (`logistic_regression_v2`) with Platt calibration |
| **Tests** | 207 automated tests, 97% coverage, lint clean (black / isort / flake8) |

<br>

---

## 🎯 About This Project

> A telecom company doesn't find out a customer is leaving until the revenue report arrives, and by then the customer is already gone. Winning them back costs far more than keeping them. A missed churner is expensive; one unnecessary retention call is cheap.

This project is the **early-warning system** that closes that gap. A production-style **Flask REST API** scores every customer's churn risk, explains *why* the score is high, recommends *what to do about it*, and then tracks whether the action actually worked. A built-in dashboard ("Retention Radar") puts it all in front of analysts and managers.

<div align="center">

```
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   7,043 Real Customers → Clean → Encode → Scale
   ─────────────────────────────────────────────────
   18 Input Features → 42 Encoded · 2 Classes (Stay/Churn)
   ─────────────────────────────────────────────────
   Predict → Explain → Recommend → Act → Measure Outcome
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

</div>

<br>

---

## ✨ What Makes This Project Stand Out

<table>
<tr>
<td width="50%" valign="top">

### 🔧 Real Data, Real Challenges
- Trained on the public **IBM Telco Customer Churn** dataset (7,043 rows)
- Only ~1 in 4 customers churns, so accuracy is misleading; models are chosen on **recall + F1**
- `TotalCharges` arrives as text with blanks and is excluded as a feature (it is just tenure × monthly charges)
- Numeric, ordinal and nominal columns handled in one sklearn pipeline

</td>
<td width="50%" valign="top">

### 🧠 Three Models Compared Fairly
- **Logistic Regression**: simple, explainable, selected
- **Random Forest**: 300 trees, depth 12
- **XGBoost**: 300 trees, depth 5
- Same stratified split, same class-imbalance treatment, 5-fold CV
- Winner's probabilities **Platt-calibrated**

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔐 Production-Style Backend
- Layered design: **routes → services → repositories → models**
- JWT auth, admin / analyst roles, real **logout** (token revocation)
- Rate limiting, CORS allow-list, security headers
- App **refuses to start** in production with weak secrets
- One consistent error envelope for every failure

</td>
<td width="50%" valign="top">

### 📊 Closed-Loop Retention
- Per-customer **key factors** in plain language
- Rule-based **recommended action** per prediction
- Campaign planner picks only customers with positive expected value
- Actions track `Kept` / `Lost` outcomes and report a real success rate

</td>
</tr>
</table>

<br>

---

## 🗺️ Full Project Journey

<div align="center">

| Stage | Focus | Key Output |
|:---:|:---|:---|
| 🔎 **Stage 1** | Data Loading & Cleaning | Telco CSV · type coercion · missing-value flags · IQR outlier report |
| 🧬 **Stage 2** | Feature Pipeline | Median/mode imputation · One-Hot · Ordinal `Contract` · StandardScaler |
| 🤖 **Stage 3** | Model Training | 3 candidates · stratified 80/20 split · 5-fold CV · weighted selection |
| ⚖️ **Stage 4** | Calibration | Platt scaling on out-of-fold predictions · Brier 0.169 → 0.139 |
| 💡 **Stage 5** | Explainability | Direction-aware key factors · recommendation rules |
| 🌐 **Stage 6** | REST API | Auth · customers · predictions · analytics · actions · imports · exports |
| 📈 **Stage 7** | Dashboard | Command center · campaign planner · action board · what-if simulator |
| 🐳 **Stage 8** | Deployment & QA | Docker Compose · Alembic migrations · 207 tests · 97% coverage |

</div>

<br>

---

## 1 · Dataset

**Real data (used to train and judge the model).** The IBM / Kaggle *Telco Customer Churn* file: 7,043 customers × 21 columns, 26.5% of customers churned. Stored in `data/raw/telco_customer_churn.csv`.

**100,000-customer sample (used to test the platform at scale).** `data/sample/customers_100k.csv` has 100,000 rows × 26 columns (20.6 MB, 26.4% churn). It is **synthetic**: `scripts/generate_dataset.py` re-samples the 7,043 real customers, adds small noise, and adds the fields a real CRM has (name, e-mail, signup date, churn date, about 1% blank values).

> ⚠️ **Important:** the model was trained on the real customers that this sample is made from. **Never use the 100k file to measure accuracy.** Use the held-out numbers in § 5.

| Feature group | Columns |
|---|---|
| Account | `tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, `signup_date`, `is_active`, `churned_at` |
| Services | `internet_service`, `phone_service`, `multiple_lines`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies` |
| Billing | `paperless_billing`, `payment_method` |
| Demographics | `gender`, `senior_citizen`, `partner`, `dependents` |
| Identity | `external_id`, `name`, `email` |

**How the data is used:** 80% train / 20% test (stratified, fixed seed). `TotalCharges` is left out of the model because it is just tenure × monthly charges; keeping it made the linear model's coefficients unreliable. Missing values are filled by the model pipeline, exactly as in training.

<div align="center">

```
7,043 rows loaded from the IBM Telco Customer Churn dataset
        ↓  Coerce TotalCharges to numeric, cast SeniorCitizen
7,043 rows · 18 input features (26.5% churn)
        ↓  Impute: median (numeric) / mode (categorical), flag columns >40% missing
        ↓  One-Hot (14 nominal) · Ordinal (Contract) · StandardScaler (numeric)
18 raw features → 42 encoded features
        ↓  Stratified 80 / 20 split
5,634 train rows · 1,409 held-out test rows
        ↓  Same fitted pipeline saved with joblib and reused when serving
Training and live predictions can never drift apart
```

</div>

<br>

---

## 2 · Topics Covered

| # | Topic | What it does (one line) | Where |
|---|---|---|---|
| 1 | **Data preparation** | Cleans types, fills missing values, scales numbers and encodes categories in one scikit-learn pipeline. | `app/ml/pipeline.py` |
| 2 | **Model selection** | Trains Logistic Regression, Random Forest and XGBoost with 5-fold CV and picks the winner on churn recall and F1, not accuracy. | `app/ml/train.py` |
| 3 | **Class imbalance** | Uses class weights because only about 1 in 4 customers churns. | `app/ml/train.py` |
| 4 | **Probability calibration** | Re-maps raw scores (Platt scaling) so "60%" really means about 60% of such customers leave. | `app/ml/calibration.py` |
| 5 | **Risk tiers** | Low below 25%, Medium below 50%, High from 50% (on calibrated probability). | `app/ml/predict.py` |
| 6 | **Explainability** | Shows the top reasons behind each score (coefficient × value) so a person can act on it. | `app/ml/predict.py` |
| 7 | **Recommendations** | Maps the top risk factor to a suggested offer using a small rules table. | `app/ml/predict.py` |
| 8 | **Bulk scoring at 100k** | Scores whole chunks at once (about 20,000 rows in 0.5 s) and imports a 100k CSV in about 30 s. | `predict_frame`, `import_service.py` |
| 9 | **Campaign ROI** | Picks only customers where expected revenue saved is higher than the offer cost; shows cost, ROI and a lift curve. | `insights_service.py` |
| 10 | **Outcome tracking** | Records whether each retention action kept or lost the customer. | `action_service.py` |
| 11 | **REST API & security** | JWT login with refresh and logout, admin / analyst roles, rate limits, CSP headers, one error format. | `app/api/`, `app/utils/` |
| 12 | **Dashboard** | Nine pages: command center, customers, explorer, campaign planner, action board, simulator, import, model quality, admin. | `dashboard/index.html` |

<details>
<summary><b>📖 Click to expand — Concepts behind these topics</b></summary>
<br>

- Why accuracy misleads on imbalanced data, and why recall and F1 on the churn class drive model selection
- How class weighting inflates raw scores, and why Platt scaling restores trustworthy probabilities
- How a Brier score and reliability bins verify that a "60%" prediction really means about 60%
- Why one saved pipeline for both training and serving prevents training/serving drift
- Why `TotalCharges` is dropped: collinearity corrupts linear-model explanations
- How risk tiers turn probabilities into an actionable call list
- Why JWT revocation, role checks and rate limiting matter for an API holding customer data
- How layered architecture keeps business logic testable and out of route handlers

</details>

<br>

---

## 3 · Project Structure

```
churn-retention-api/
├── app/
│   ├── api/            # routes (auth, customers, predictions, analytics, actions, imports, exports, admin, model)
│   ├── services/       # business logic (analytics, insights, imports, predictions, ...)
│   ├── repositories/   # all database queries
│   ├── models/         # SQLAlchemy tables
│   ├── schemas/        # request / response validation (also feeds the /docs page)
│   ├── ml/             # pipeline, training, calibration, prediction, saved model (artifacts/)
│   └── utils/          # errors, auth decorators, cache, logging
├── dashboard/index.html        # the whole web app (one file)
├── data/
│   ├── raw/telco_customer_churn.csv        # real dataset (7,043 rows)
│   └── sample/customers_100k.csv           # synthetic 100,000-row sample
├── migrations/                 # Alembic (only needed for PostgreSQL)
├── scripts/generate_dataset.py # makes the 100k sample
├── screenshots/                # images used in this README
├── tests/                      # unit + integration tests
├── requirements.txt            # run-time packages
├── requirements-dev.txt        # test + code-style packages
├── .env.example                # optional settings
└── wsgi.py                     # entry point
```

The code follows one rule: **routes → services → repositories → models.** Routes only handle HTTP, services hold the logic, repositories hold the queries.

<div align="center">

```
Dashboard / API client
        ↓  HTTPS + JWT
   app/api/           ← thin routes (flask-smorest Blueprints)
        ↓  Marshmallow validation
   app/services/      ← business logic
        ↓
   app/repositories/  ← the ONLY layer that builds ORM queries
        ↓
   app/models/        ← SQLAlchemy 2.x  →  SQLite / PostgreSQL
        ┊
   app/ml/predict.py  ← one predict() path for API, batch, imports and seeding
```

</div>

<br>

---

## 4 · Dashboard Screenshots

Sign in at `http://127.0.0.1:5000/dashboard` (all screenshots below use the 100,000-customer sample).

### 🔐 Sign in

![Sign in](screenshots/01-login.png)
*Figure 1 · Sign-in page. Accounts are created by an admin, nothing is pre-filled.*

### 🧭 Command center

![Command center](screenshots/02-command-center.png)
*Figure 2 · Monthly revenue at risk, a 100-block risk bar of all active customers (1 block = 1%), weekly churn trend and the "who to call first" list.*

### 👥 Customers

![Customers](screenshots/03-customers.png)
*Figure 3 · Search, filter and sort across about 73,600 active customers with page controls and CSV export.*

![Customer detail](screenshots/04-customer-detail.png)
*Figure 4 · Customer detail: churn chance gauge, the reasons behind the score (red pushes toward leaving, green toward staying), profile and actions.*

### 🧩 Segment explorer

![Segment explorer](screenshots/05-segment-explorer.png)
*Figure 5 · Group customers by contract, tenure, bill, add-ons and more to see where revenue at risk is concentrated.*

### 💰 Campaign planner

![Campaign planner](screenshots/06-campaign-planner.png)
*Figure 6 · Set offer cost, success rate, months and budget; the planner shows who to contact, total cost, net gain, ROI and the lift curve.*

### 🔮 What-if simulator

![What-if simulator](screenshots/07-what-if-simulator.png)
*Figure 7 · Change a customer's contract, price or add-ons and watch the churn chance and its drivers move. Nothing is saved.*

### 📋 Action board

![Action board](screenshots/08-action-board.png)
*Figure 8 · Drag-and-drop board (to do / in progress / finished) with due dates and Kept / Lost outcomes. The top cards show the real success rate.*

### 📥 Import data

![Import data](screenshots/09-import-data.png)
*Figure 9 · Drag-and-drop CSV import with live progress, a row-by-row error report and a downloadable template.*

### 🧠 Model quality

![Model quality](screenshots/10-model-quality.png)
*Figure 10 · Held-out recall, precision, F1 and ROC-AUC, calibration table, ROC curves and confusion matrix.*

### 🌙 Dark mode

![Dark mode](screenshots/11-dark-mode.png)
*Figure 11 · Same command center in dark mode (switch with the moon button). Ctrl + K opens search from any page.*

<br>

---

## 5 · Results

All model numbers come from the **1,409 held-out customers** the model never saw (5,634 train / 1,409 test). The model was chosen for churn **recall and F1**, because missing a customer who will leave costs more than one extra offer.

### 5a · Model comparison (held-out test set)

<div align="center">

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Logistic Regression** | 74.0% | 50.7% | **77.5%** | **61.3%** | **83.8%** | ⭐ **Selected** |
| XGBoost | 74.7% | 51.7% | 71.9% | 60.2% | 82.6% | 🥈 Close second |
| Random Forest | 77.4% | 56.7% | 62.3% | 59.4% | 82.7% | 🥉 Highest accuracy, lowest recall |

</div>

Confusion matrix of the selected model: **753** stayed and predicted stay, **282** stayed but flagged, **84** churners missed, **290** churners caught.

> 💡 **Key insight:** the three models are within about 1.3 ROC-AUC points of each other. Logistic Regression wins on *equal ranking quality + best recall + simplest to explain*, not by a landslide. Selection score = `0.4 × recall + 0.4 × F1 + 0.2 × ROC-AUC` on the churn class.

### 5b · Are the probabilities honest? (calibrated, held-out)

Average predicted churn **26.7%** vs actual **26.5%** (Brier score **0.139**, down from 0.169 before calibration; raw uncalibrated mean was 41.6%).

| Risk tier | Share of customers | Predicted churn | Actually churned | Share of all churners |
|---|---|---|---|---|
| 🔴 High (50%+) | 22.8% | 63.9% | 64.2% | 55.1% |
| 🟠 Medium (25–50%) | 19.7% | 37.2% | 32.5% | 24.1% |
| 🟢 Low (below 25%) | 57.6% | 8.4% | 9.6% | 20.9% |

> 📞 Contacting only the **High** tier reaches more than half of all churners while touching about a fifth of the customer base.

### 5c · Scale test (100,000-customer synthetic sample, SQLite, one laptop-class machine)

| Task | Result |
|---|---|
| Import 100,000 customers and score 73,610 active ones | about 29 seconds |
| Score 20,000 rows (vectorised) | 0.5 seconds |
| Every analytics endpoint (overview, segments, trends, risk, campaign, explorer) | under 1 second |
| Customers with risk **High / Medium / Low** (active) | 10.2% / 18.3% / 71.5% |
| Monthly revenue at risk | about $615,000 of $4.51M (13.7%) |

These scale numbers show speed and the dashboard, not model accuracy (see the note in § 1).

### 5d · Business insight

- **Month-to-month contracts hold essentially all the risk:** 23% of active month-to-month customers are High risk, against under 1% on one- and two-year contracts.
- **Best first move:** offer month-to-month customers a 12-month contract discount, starting with the highest expected value (churn chance × monthly charge).
- **Do not trust the explanations as causes.** They show what the model leaned on, which is a good prompt for a conversation, not proof of why someone leaves.

### 5e · What the real data says

<div align="center">

| Factor | Churn rate |
|:---|:---:|
| Month-to-month / one-year / two-year contract | **42.7%** / 11.3% / 2.8% |
| Fiber / DSL / no internet | **41.9%** / 19.0% / 7.4% |
| Electronic check vs other payment methods | **45.3%** vs 15–19% |
| Tenure 0–12 months / 49–72 months | **47.4%** / 9.5% |
| No tech support / tech support | **41.6%** / 15.2% |
| Month-to-month **and** fiber | **54.6%** |

*These are correlations, not proof of cause. Confirming that an incentive reduces churn needs an A/B test.*

</div>

<br>

---

## 6 · Quick Start

Requires Python 3.12 (3.10+ works). No database or Redis needed.

```bash
# 1. install
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # Windows: copy .env.example .env

# 2. create the database, the model check and an admin user
python -m flask --app wsgi init-db
python -m app.ml.ensure_model       # uses the bundled model; retrains only if it is missing
python -m flask --app wsgi create-admin --email admin@example.com --password 'Admin@12345'

# 3. load the 100,000 customers (about 30 s) and, optionally, demo actions
python -m flask --app wsgi import-csv data/sample/customers_100k.csv
python -m flask --app wsgi seed-actions

# 4. run
python -m flask --app wsgi run      # open http://127.0.0.1:5000/dashboard
```

Sign in with `admin@example.com` / `Admin@12345` (**change this password for any real use**).

| Command | Purpose |
|---|---|
| `python -m flask --app wsgi seed-demo --count 300` | Small demo set from the real Telco file (with billing history) |
| `python -m flask --app wsgi train` | Retrain and compare the three models |
| `python scripts/generate_dataset.py --rows 100000` | Rebuild the 100k sample (any size, `--as-of YYYY-MM-DD`, `--seed N`) |
| `http://127.0.0.1:5000/docs` | Interactive API documentation (Swagger) |

**Optional settings** live in `.env` (see `.env.example`): `DATABASE_URL` for PostgreSQL (then run `flask db upgrade` instead of `init-db`), `REDIS_URL` for shared caching, token lifetimes, rate limits and the risk thresholds. In production (`FLASK_ENV=production`) the app refuses to start unless `SECRET_KEY` and `JWT_SECRET_KEY` are long random values.

<details>
<summary><b>🐳 Click to expand — Docker (PostgreSQL + Redis)</b></summary>
<br>

```bash
cp .env.example .env          # set real SECRET_KEY / JWT_SECRET_KEY (32+ chars)
docker compose up --build
docker compose exec web flask create-admin --email you@example.com --password "a-strong-password"
```

The compose stack runs web + PostgreSQL 16 + Redis 7 with health checks; database and Redis ports are not exposed to the host.

</details>

<br>

---

## 7 · API Summary

All endpoints return JSON; every error uses `{"error": {"code", "message", "details"}}`. Full details are on `/docs` (OpenAPI spec at `/openapi.json`).

| Area | Endpoints | Who |
|---|---|---|
| 🔑 Auth | `POST /auth/login`, `/auth/refresh`, `/auth/logout`, `GET /auth/me`, `POST /auth/register` (admin) | everyone / admin |
| 👥 Customers | `GET/POST /customers`, `GET/PUT/DELETE /customers/<id>`, `GET/POST /customers/<id>/transactions` | analyst, admin (delete: admin) |
| 🎯 Predictions | `POST /predict` (stored customer or raw features), `POST /predict/batch`, `GET /predictions` | analyst, admin |
| 📊 Analytics | `GET /analytics/overview`, `/segments`, `/trends`, `/risk-distribution`, `/revenue-at-risk`, `/high-risk`, `/campaign`, `/explorer`, `/outcomes` | analyst, admin |
| 📝 Actions | `GET/POST /actions`, `POST /actions/bulk`, `GET/PUT/DELETE /actions/<id>` | analyst, admin (delete: admin) |
| 📥 Import | `POST /imports` (CSV), `GET /imports`, `GET /imports/<id>`, `GET /imports/template.csv` | upload: admin |
| 📤 Export | `GET /export/customers.csv`, `/export/predictions.csv`, `/export/high-risk.csv` | analyst, admin |
| 🧠 Model / Admin | `GET /model/info`, `/admin/users`, `/admin/logs`, `/admin/activity`, `GET /health` | analyst, admin |

**Example: real-time prediction** (`POST /predict`)

```json
{
  "churn_probability": 0.791,
  "risk_level": "High",
  "model_version": "logistic_regression_v2",
  "key_factors": [
    {"feature": "tenure (short)", "impact": 0.9061, "direction": "increases churn risk"},
    {"feature": "InternetService=Fiber optic", "impact": 0.7618, "direction": "increases churn risk"}
  ],
  "recommended_action": "Schedule a proactive onboarding check-in call"
}
```

**CSV import columns:** required `email, name, tenure_months, monthly_charges, contract_type` (`month-to-month`, `one-year`, `two-year`). Optional: `external_id, signup_date, total_charges, plan, is_active, churned_at` and the service columns (`internet_service`, `tech_support`, `payment_method`, ...). Download the template from the dashboard or `GET /imports/template.csv`.

<br>

---

##  8 · Testing

```bash
pip install -r requirements-dev.txt
python -m pytest --cov=app --cov-report=term-missing          # in-memory SQLite
black --check --line-length 120 app tests scripts && flake8 app tests scripts
```

**207 tests, 97% coverage.** They cover the ML pipeline and training, single and vectorised scoring (checked row by row to give identical results), every API endpoint (success, validation errors, permissions), the CSV import (good, bad and duplicate rows, skip vs update), campaign and explorer math, bulk actions, the CLI and rate limiting. A tiny model is built inside the tests, so they do not depend on the trained one.

<br>

---

<br>

---

## 9 · Tools Used & Requirements

| Purpose | Tools |
|---|---|
| Web API | Flask 3, flask-smorest (OpenAPI / Swagger), marshmallow |
| Database | SQLAlchemy 2, Flask-SQLAlchemy, Alembic / Flask-Migrate, SQLite (default) or PostgreSQL |
| Security | Flask-JWT-Extended, bcrypt, Flask-Limiter, Flask-Talisman, Flask-Cors |
| Caching | Flask-Caching (in-process, or Redis if `REDIS_URL` is set) |
| Machine learning | scikit-learn, XGBoost, SHAP, pandas, numpy, joblib |
| Dashboard | HTML, vanilla JavaScript, Chart.js (CDN), Google Fonts (Manrope, Sora) |
| Server / DevOps | Gunicorn, Docker (multi-stage), docker-compose |
| Testing & style | pytest, pytest-cov, black, isort, flake8 |

All Python packages with pinned versions are listed in `requirements.txt` (run-time) and `requirements-dev.txt` (tests and code style).

<br>

---

## 10 · Known Limitations

- **The 100k file is synthetic.** It is for speed and demo only; judge the model on the held-out numbers in § 5.
- **Explanations are conditional, not causal.** Use them as hints for a conversation.
- **The real dataset is a one-time snapshot.** It has no dates, so real trend forecasting or drift detection is not possible with it. The weekly trend chart uses the dates in the synthetic sample.
- **Campaign success rate is your assumption** until enough Kept / Lost outcomes are recorded on the action board.
- **Fairness:** `gender` and `SeniorCitizen` are model inputs. Check that offers driven by them are acceptable where you operate.
- **Retraining is manual** (`flask train`). There is no scheduled retraining or drift alert.
- **CSV imports run in a background thread inside the web process**, so a worker restart during a large import can leave the job unfinished. A job queue would be the next step at larger scale.
- **PostgreSQL** is supported but the newest migration was not run against a live PostgreSQL server in this build.
- **Production hardening:** set `FLASK_ENV=production` and real secrets, and change the demo admin password. Only load model files (`.pkl`) from a source you trust.

<br>


---

## 👨‍💻 Author

<div align="center">

### **Ayush Isamaliya**
*Data Science & Aspiring ML Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-isamaliya16-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/isamaliya16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Isamaliya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-isamaliya-686533312/)

</div>

<br>

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:2E86C1,100:0B3D6B&height=130&section=footer)

*Made with ❤️ | Churn & Retention Platform | Flask + scikit-learn + Calibrated Probabilities | Customer Retention Analytics*

</div>

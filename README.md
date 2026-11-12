# Claim Outcome Prediction — End-to-End ML Mini Build

Predicts whether an insurance claim will be **Paid**, **Partially_Paid** or **Denied**, and serves the model through a FastAPI endpoint.

## Project layout
| File | Purpose |
|---|---|
| `generate_data.py` | Simulates 450 claims (~70/20/10 imbalance) -> `claims.csv` |
| `features.py` | Leakage-safe feature engineering (sklearn transformer) |
| `train.py` | Stratified split, training, evaluation, saves `model.joblib` + `metrics.txt` |
| `app.py` | FastAPI service: `POST /predict`, `GET /appeal-guidance` (bonus) |
| `appeal.py` | Bonus: keyword retrieval of appeal snippets |
| `test_api.py` | Tests: valid request, missing field, text amount, unseen payer |

## How to run on Windows (PowerShell)
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_data.py
python train.py
python -m pytest -q
python -m uvicorn app:app --port 8000
```
Then open http://127.0.0.1:8000/docs and try `POST /predict`.

## How to run on Linux/Mac (commands)
```bash
# 1. Create environment and install
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate the dataset
python generate_data.py

# 3. Train + evaluate + save the pipeline
python train.py

# 4. Start the API
uvicorn app:app --reload --port 8000

# 5. Call it (new terminal)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"payer":"Aetna","provider":"Dr. Rao","cpt_code":"99213","diagnosis_code":"M54.5","billed_amount":220.0,"patient_age":45}'

# 6. Bad request -> HTTP 400
curl -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"payer":"Aetna","billed_amount":"abc"}'

# 7. Bonus
curl "http://127.0.0.1:8000/appeal-guidance?denial_reason=missing_auth"

# 8. Tests
pytest -q
```
Interactive docs are at http://127.0.0.1:8000/docs.

## Step-by-step explanation

**Step 0 — Data (`generate_data.py`).** Random rows with a fixed seed. Hidden tendencies (e.g. Cigna denies more, 99215 is more often partially paid, over-billed claims and seniors are riskier) give the model real signal. Result: 66.4% Paid / 22.7% Partial / 10.9% Denied.

**Step 1 — Prepare data (`train.py`, `features.py`).**
- `train_test_split(..., stratify=y)` gives an 80/20 split that keeps the class mix on both sides (train 66/23/11, test 67/22/11).
- The split happens **first**, then features are learned from training rows only.
- Engineered features (5):
  - `payer_denial_rate`, `provider_denial_rate`: smoothed denial rate per payer/provider
  - `cpt_partial_rate`: partial-payment rate per CPT code
  - `billed_vs_cpt_median`: billed amount relative to that CPT's typical amount
  - `is_senior`: age >= 65
- **Leakage prevention:** these live in a `ClaimFeatures` transformer inside the sklearn `Pipeline`. `fit()` only ever sees training rows; test rows and API requests only get lookups. Unseen payers fall back to the global rate.

**Step 2 — Train (`train.py`).** Pipeline = `ClaimFeatures` -> `ColumnTransformer` (one-hot categoricals, numerics passed through) -> `RandomForestClassifier` with `class_weight="balanced_subsample"`. Saved as one file, `model.joblib`, so preprocessing and model can't drift apart.

**Step 3 — Evaluate.** Per-class precision/recall/F1 and a confusion matrix on the held-out test set (90 rows).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Paid | 0.686 | 0.800 | 0.738 | 60 |
| Partially_Paid | 0.182 | 0.100 | 0.129 | 20 |
| Denied | 0.222 | 0.200 | 0.211 | 10 |

Overall accuracy 0.578. A "always Paid" baseline gets ~0.67, so **accuracy alone would be misleading** — this model is currently *worse* than that baseline on accuracy, but it does try to flag minority classes.

Confusion matrix (rows = actual):
```
                 pred_Paid  pred_Partial  pred_Denied
actual_Paid          48          7             5
actual_Partial       16          2             2
actual_Denied         6          2             2
```
(Exact numbers may vary slightly by library version.)

**Weakest class.** The model is weakest at Partially_Paid (it caught only 2 of 20, recall 0.10) and then Denied (2 of 10, recall 0.20). In most of these cases it simply guessed "Paid". I think this is because the data is small (only about 45 Denied rows in total, and just 10 in the test set) and the pattern I planted in the simulated data is weak. **Idea to improve:** collect more data and richer features (like prior-authorization status), lower the probability threshold used to flag Denied so more denials are caught, and check results with cross-validation instead of one small split.

**Step 4 — Serve (`app.py`).** `POST /predict` validates input with pydantic; any missing/invalid field returns **HTTP 400** with a clear message instead of a crash. `"Dr. Rao"` is normalised to the training format `"Dr.Rao"`. Response matches the spec (`predicted_status` + `probabilities`).

**Bonus (`appeal.py`).** 7 appeal-guidance snippets retrieved by keyword overlap on `denial_reason`; unknown reasons return a "escalate to human" message.

### Reducing hallucination risk when an LLM drafts the appeal letter
1. Pass the retrieved snippet as the only allowed source and instruct the model to use nothing else.
2. Supply facts (patient, dates, amounts, claim ID) as structured fields; never let the model invent them.
3. Instruct it to output "INSUFFICIENT INFORMATION" rather than guess missing details.
4. Use a fixed letter template; the LLM only fills specific slots.
5. Low temperature and structured (JSON) output validated against a schema.
6. Post-check: confirm every number, date and policy citation in the draft appears in the inputs.
7. Never cite policy numbers/laws not present in the retrieved text.
8. A human biller reviews and approves before anything is sent.

## Assumptions and shortcuts (because of the 90-minute limit)
- The data is simulated by me, so the results show how the model behaves on my fake data, not on real insurance claims.
- I used one 80/20 split. With only 10 Denied rows in the test set, those scores can change a lot from one run to another. Cross-validation would be more reliable.
- The denial-rate features are calculated from training rows only (no leakage), but each training row also counts its own label in its category's rate. I reduced this with smoothing; out-of-fold encoding would be the proper fix.
- I did not tune the model settings, and the API has no login, logging or monitoring.
- I treat "Dr. Rao" and "Dr.Rao" as the same provider by removing spaces.

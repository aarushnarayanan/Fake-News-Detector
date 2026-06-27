# VeriScope

RoBERTa-based fake news detector. Paste text or a URL → get a REAL/FAKE label, confidence score, and highlighted suspicious sentences.

**Stack:** React + Vite (frontend), FastAPI (backend), PostgreSQL (history), RoBERTa fine-tuned on LIAR dataset.

---

## Quick Start

```bash
cp .env.example .env       # fill in POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
docker-compose up --build
```

- Frontend: http://localhost:80
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

> **Model files** (`models/roberta_text_v3/`) are not in git. Download or train them first — see [Training](#training).

---

## Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `POSTGRES_USER` | DB username |
| `POSTGRES_PASSWORD` | DB password |
| `POSTGRES_DB` | DB name (e.g. `fakenews_db`) |
| `VITE_API_URL` | Backend URL seen by the browser (default: `http://localhost:8000`) |
| `CORS_ORIGINS` | Comma-separated allowed origins for CORS |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Run model on text. Body: `{"text": "..."}` |
| `POST` | `/scrape` | Scrape URL then analyze. Body: `{"url": "..."}` |
| `GET` | `/history` | Paginated past predictions |
| `POST` | `/feedback` | Submit correction for a prediction |

---

## Training

Models live in `models/`. Three versions exist (`roberta_text`, `roberta_text_v2`, `roberta_text_v3`); the backend loads `v3` by default.

```bash
pip install -r requirements-ml.txt
python models/train.py          # trains v1
bash retrain_v3.sh              # trains v3 (used in prod)
python models/calibrate.py      # temperature-scales the model after training
```

Evaluation scripts are in `predict.py_test/`.

---

## Running Without Docker

```bash
# Backend
pip install -r backend/requirements.txt -r requirements-ml.txt
uvicorn backend.main:app --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Set `VITE_API_URL=http://localhost:8000` in your shell or a `.env.local` in `frontend/`.

---

## Project Layout

```
backend/        FastAPI app, DB models, Dockerfile
frontend/       React/Vite UI, Dockerfile
models/         Training, eval, predict scripts + saved model dirs
predict.py_test/ Offline eval scripts and outputs
validation_set_builder/ Tools for building the validation dataset
```

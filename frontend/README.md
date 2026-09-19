# MLReady - Frontend

React dashboard for the MLReady ML Readiness & Risk Analyzer. Talks to the
FastAPI backend in `../backend`.

## Setup

```
npm install
cp .env.example .env   # adjust VITE_API_BASE_URL if your backend isn't on :8000
npm run dev
```

Open http://localhost:5173. Make sure the backend is running first (see
`../backend/README` equivalent instructions in the project docs) - either
`uvicorn main:app --reload` (needs MongoDB) or `python demo_server.py`
(in-memory, no MongoDB needed, good for a quick look).

## Pages

Login / Register -> Dashboard (all your datasets) -> Upload -> per-dataset
pipeline: Overview -> Data Quality -> ML Risk Analysis -> Cleaning -> Model
Comparison -> Analysis History. The sidebar's pipeline links light up once
you've opened a dataset.

Model Comparison will show "not wired up yet" until the Phase 5 backend
routes (`POST /datasets/{id}/train`, `GET /datasets/{id}/model-results`)
exist - the page itself is complete and needs no changes when they land.

## Build

```
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
npm run lint       # oxlint
```

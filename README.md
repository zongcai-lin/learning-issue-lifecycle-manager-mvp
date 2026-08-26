# Learning Issue Lifecycle Manager (LILM)

LILM is an early-stage prototype that turns an AI learning conversation into structured goal threads and issue states. It is intended to help learners see what they tried, where they became blocked, and what remains unresolved instead of losing that context inside a long chat transcript.

> **Work in progress:** this repository is being shared as an interim portfolio
> checkpoint. The core multi-goal analysis pipeline is implemented and tested,
> while runtime hardening, public regression fixtures, and the final MVP
> readiness review are still planned work.

## What you can review today

| Area | Status | Notes |
| --- | --- | --- |
| Conversation parsing | Implemented | Deterministic parsing for common pasted transcript formats. |
| Goal detection | Implemented | Stage 1 identifies multiple learning goals and assigns turns to each goal. |
| Per-goal issue extraction | Implemented | Stage 2 independently extracts issues, lifecycle states, evidence, and next actions for each goal. |
| API and browser UI | Prototype | FastAPI endpoint and lightweight frontend are wired for local use. |
| Runtime and API error handling | Planned (LILM-5) | Lazy client initialization, user-facing failure handling, and browser-path verification remain outstanding. |
| Public regression fixtures | Planned (LILM-6) | Current tests use synthetic data; a curated public fixture set is still outstanding. |
| Final Step 6.2 readiness audit | Planned (LILM-7) | This checkpoint is not the final MVP release. |

## Current capabilities

- Accepts a pasted learning conversation through a FastAPI endpoint.
- Uses a staged OpenAI analysis pipeline with Pydantic structured outputs: Stage 1
  detects Goal Threads, then Stage 2 extracts Issues independently for each goal.
- Represents multiple learning goals, issue lifecycle states, evidence, and next actions.
- Renders goal summaries and issue cards in a lightweight browser interface.
- Includes a deterministic conversation parser and unit tests for common transcript formats.

## Status and limitations

This repository contains an in-progress MVP, not a production service. The staged
parser and analyzer are wired into `/analyze`, but runtime/API failures have not yet
received the planned LILM-5 hardening. There is currently no authentication,
persistence, rate limiting, or production deployment configuration.

The text submitted to `/analyze` is sent to the configured OpenAI API. Do not submit confidential, personal, or regulated information. No real user conversations or API credentials are committed to this repository.

## Tech stack

- Python, FastAPI, Pydantic
- OpenAI API structured output
- Vanilla HTML, CSS, and JavaScript

## Local setup

1. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy the environment template and add your own API key:

   ```bash
   cp .env.example .env
   ```

3. Start the API:

   ```bash
   uvicorn backend.main:app --reload
   ```

4. In another terminal, serve the frontend:

   ```bash
   python -m http.server 5500 --directory frontend
   ```

5. Open `http://localhost:5500`.

The development CORS configuration accepts the local frontend origins defined in `backend/main.py`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Repository privacy

`.env`, virtual environments, caches, and local agent-workflow files are excluded through `.gitignore`. Use `.env.example` only as a template and never commit a real API key.

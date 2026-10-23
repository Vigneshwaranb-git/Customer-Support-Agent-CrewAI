# CrewAI Support Crew

KB (RAG) agent -> web-search agent (only if KB fails) -> Google Docs logging agent.

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, fill in keys.
3. Google Docs: enable the Google Docs API in Google Cloud, create a service account,
   download its JSON key as `service_account.json`, create a Google Doc, and **share it with
   the service account's email as Editor**. Put the doc ID in `.env`.
4. Add your support docs (.md/.txt) to `data/kb/`, then run `python ingest.py`.
5. `python main.py "Do you ship to Canada?"`

"""
Customer_Support_Agent.py
-------------------------
CrewAI customer support crew in a single file:

  1. Support agent   -> answers from the knowledge base via a RAG tool
  2. Web researcher  -> runs ONLY if the knowledge-base answer is unsatisfactory
  3. Logger agent    -> appends (time, question, answer) to a Google Sheet

Usage:
  python Customer_Support_Agent.py --ingest              # index files in data/kb
  python Customer_Support_Agent.py "How do I return an item?"
"""
import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Type

import chromadb
from chromadb.utils import embedding_functions
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pydantic import BaseModel, Field

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
KB_DIR = "data/kb"
CHROMA_PATH = "chroma_db"
COLLECTION = "support_kb"
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ----------------------------------------------------------------------------
# RAG: vector store helpers + ingestion
# ----------------------------------------------------------------------------
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk(text: str, size: int = 800, overlap: int = 100):
    start = 0
    while start < len(text):
        yield text[start:start + size]
        start += size - overlap


def ingest_knowledge_base():
    """Index every .md/.txt file under data/kb into ChromaDB."""
    col = get_collection()
    ids, docs, metas = [], [], []
    for path in Path(KB_DIR).glob("**/*"):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        for i, c in enumerate(_chunk(path.read_text(encoding="utf-8"))):
            ids.append(hashlib.md5(f"{path}-{i}".encode()).hexdigest())
            docs.append(c)
            metas.append({"source": path.name})
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} chunks from {KB_DIR}.")


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
class KBSearchInput(BaseModel):
    query: str = Field(..., description="The customer's question, rephrased as a search query.")


class KnowledgeBaseSearchTool(BaseTool):
    name: str = "Knowledge Base Search"
    description: str = (
        "Searches the company's internal support knowledge base (policies, FAQs, product docs). "
        "Always use this first. Returns the most relevant passages with similarity scores, "
        "or NO_RELEVANT_RESULTS if nothing matches."
    )
    args_schema: Type[BaseModel] = KBSearchInput

    def _run(self, query: str) -> str:
        min_sim = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
        res = get_collection().query(query_texts=[query], n_results=4)
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]

        hits = []
        for doc, meta, dist in zip(docs, metas, dists):
            similarity = 1 - dist  # cosine distance -> similarity
            if similarity >= min_sim:
                hits.append(f"[source: {meta['source']} | similarity: {similarity:.2f}]\n{doc}")
        return "\n\n---\n\n".join(hits) if hits else "NO_RELEVANT_RESULTS"


class LogInput(BaseModel):
    question: str = Field(..., description="The exact question the customer asked.")
    answer: str = Field(..., description="The final answer given to the customer.")
    source: str = Field("knowledge_base", description="'knowledge_base' or 'web'.")


class GoogleSheetsLogTool(BaseTool):
    name: str = "Log to Google Sheets"
    description: str = (
        "Appends one row (timestamp, question, answer, source) to the support log Google Sheet."
    )
    args_schema: Type[BaseModel] = LogInput

    def _run(self, question: str, answer: str, source: str = "knowledge_base") -> str:
        creds = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"], scopes=GOOGLE_SCOPES
        )
        sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
        sheet_id = os.environ["GOOGLE_SHEET_ID"]

        # Find the first tab's name so this works whatever it is called.
        meta = sheets.spreadsheets().get(
            spreadsheetId=sheet_id, fields="sheets.properties.title"
        ).execute()
        tab = meta["sheets"][0]["properties"]["title"]

        # Timestamp is added by code (not the LLM) so it is always accurate.
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheets.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range=f"'{tab}'!A:D",
            valueInputOption="RAW",  # RAW stores text as-is (no formula interpretation)
            insertDataOption="INSERT_ROWS",
            body={"values": [[ts, question, answer, source]]},
        ).execute()
        return f"Logged row at {ts}"


# ----------------------------------------------------------------------------
# Crew
# ----------------------------------------------------------------------------
class SupportAnswer(BaseModel):
    answer: str
    resolved: bool  # True only if the answer fully and confidently addresses the question
    source: str     # "knowledge_base" or "web"


def kb_failed(output: TaskOutput) -> bool:
    """Condition for the web task: run only if the KB answer was not satisfactory."""
    parsed = output.pydantic
    return parsed is None or not parsed.resolved


def build_crew() -> Crew:
    llm = LLM(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.2)

    # ---- Agents ----
    support_agent = Agent(
        role="Customer Support Specialist",
        goal="Answer customer questions accurately using ONLY the company knowledge base.",
        backstory="You are a friendly, precise support rep. You never guess; if the knowledge "
                  "base doesn't clearly answer the question, you mark it unresolved.",
        tools=[KnowledgeBaseSearchTool()],
        llm=llm,
        verbose=True,
    )
    web_researcher = Agent(
        role="Web Research Specialist",
        goal="Find a correct, customer-friendly answer on the web when the knowledge base falls short.",
        backstory="You search reputable sources, cross-check them, and write concise answers.",
        tools=[SerperDevTool()],
        llm=llm,
        verbose=True,
    )
    logger_agent = Agent(
        role="Support Log Keeper",
        goal="Record every handled customer question and its final answer in Google Sheets.",
        backstory="You are meticulous about record-keeping and always log using the tool.",
        tools=[GoogleSheetsLogTool()],
        llm=llm,
        verbose=True,
    )

    # ---- Tasks ----
    answer_task = Task(
        description=(
            "Customer question: {question}\n\n"
            "Search the knowledge base and answer. Set resolved=true only if the retrieved "
            "passages clearly answer the question. If the tool returns NO_RELEVANT_RESULTS or "
            "the passages are only loosely related, set resolved=false."
        ),
        expected_output="A SupportAnswer with answer, resolved flag, and source='knowledge_base'.",
        agent=support_agent,
        output_pydantic=SupportAnswer,
    )
    web_task = ConditionalTask(
        description=(
            "The knowledge base could not fully answer: {question}\n"
            "Search the web, verify across sources, and write a clear, accurate answer for the customer."
        ),
        expected_output="A SupportAnswer with the web-based answer, resolved=true, source='web'.",
        agent=web_researcher,
        condition=kb_failed,
        output_pydantic=SupportAnswer,
    )
    log_task = Task(
        description=(
            "Customer question: {question}\n\n"
            "From the previous tasks, take the final answer: use the web answer if one exists, "
            "otherwise the knowledge base answer. Call the 'Log to Google Sheets' tool with the "
            "question, that final answer, and its source. Then output ONLY the final answer text "
            "for the customer."
        ),
        expected_output="The final answer text shown to the customer.",
        agent=logger_agent,
        context=[answer_task, web_task],
    )

    return Crew(
        agents=[support_agent, web_researcher, logger_agent],
        tasks=[answer_task, web_task, log_task],
        process=Process.sequential,
        verbose=True,
    )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrewAI customer support agent")
    parser.add_argument("question", nargs="*", help="Customer question")
    parser.add_argument("--ingest", action="store_true", help="Index data/kb into the vector store")
    args = parser.parse_args()

    if args.ingest:
        ingest_knowledge_base()
    else:
        question = " ".join(args.question) or "How do I return an item?"
        result = build_crew().kickoff(inputs={"question": question})
        print("\n=== ANSWER TO CUSTOMER ===\n", result.raw)

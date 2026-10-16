# Start here — run ePolicies on your laptop with your own .md files

You need: your policy `.md` folder, a Google API key (with access to `gemini-embedding-001` and a
generation model such as `gemini-3.5-flash`), and about 30 minutes. Full details: [README.md](README.md).

## 1. Install the tools (once)

| | Windows | Mac |
|---|---|---|
| Python 3.12 | python.org installer — tick **"Add python.exe to PATH"** | `brew install python@3.12` |
| Node.js 20+ (LTS) | nodejs.org installer | `brew install node` |
| PostgreSQL + pgvector | **Easiest:** Docker Desktop, then step 3 option A. Without Docker: PostgreSQL 17 installer from postgresql.org + pgvector built per github.com/pgvector/pgvector ("Windows" section, needs Visual Studio C++ tools) | `brew install postgresql@17 pgvector` then `brew services start postgresql@17` |

If your company blocks installs, ask IT for Python, Node.js and a PostgreSQL database **with the pgvector
extension** (they give you host, port, database name, user and password).

## 2. Get the project

```
git clone https://github.com/charan4590/VeritasAI-Multi-Agent-Research-Intelligence-Platform.git
cd VeritasAI-Multi-Agent-Research-Intelligence-Platform
git checkout epolicies-rag
cd epolicies-rag
```

## 3. Start a database

- **A — Docker (Windows or Mac):** `docker compose up -d db` → host `localhost`, port `5432`, user `postgres`, password `postgres`, database `epolicies`.
- **B — Installed PostgreSQL:** use the user/password you chose during install (the setup wizard can create the `epolicies` database for you).
- **C — Company server:** use the details from IT.

## 4. Install the backend and run the setup wizard

Windows (PowerShell):

```
py -3.12 -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd backend
python -m app.cli setup
```

Mac:

```
python3.12 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m app.cli setup
```

(If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

The wizard asks for:

1. **The folder with your .md files** — paste its full path (Windows: Shift+right-click the folder → *Copy as path*; Mac: Option+right-click → *Copy as Pathname*). It counts the files so you can confirm (~235). The files stay where they are and are never changed.
2. **Your Google API key** (hidden while typing) and the answer model (default `gemini-3.5-flash`).
3. **Database details** (host, port, name, user, password).

It writes `epolicies-rag/.env`, creates the tables, tests everything, and offers to index your documents.

## 5. Check and index

```
python -m app.cli doctor          # every line should say [ OK ]
python -m app.cli ingest          # reads, chunks and embeds all .md files (safe to re-run)
python -m app.cli status          # how many documents/chunks are indexed
```

If `ingest` stops (quota, network), run `python -m app.cli resume` later — finished work is not repeated.

## 6. Start the app (two terminals)

Terminal 1 — API (keep the virtual environment active, inside `epolicies-rag/backend`):

```
uvicorn app.main:app --port 8000
```

Terminal 2 — website (inside `epolicies-rag/frontend`):

```
npm ci
npm run dev
```

Open **http://localhost:3000**. The *Knowledge base* page lists your documents; ask a question in *Chat*.

## When something is wrong

Run `python -m app.cli doctor` — it says what failed and how to fix it. Common ones:

| Problem | Fix |
|---|---|
| `No .md files found` | Wrong folder; run `python -m app.cli setup` again or edit `MARKDOWN_DIR` in `.env`. |
| `connection refused` | PostgreSQL/Docker is not running, or wrong host/port. |
| `password authentication failed` | Wrong database user/password. |
| `pgvector is NOT installed` | Use the Docker option, or ask IT to install pgvector on the server. |
| `gemini_auth` | Key is wrong or not enabled for the Gemini API. |
| `gemini_quota` / 429 | Wait, or set another `GEMINI_MODEL` in `.env` (e.g. `gemini-3.5-flash-lite`). |
| Website says "API offline" | Terminal 1 is not running. |

When you add, change or delete `.md` files later, just run `python -m app.cli ingest` again.

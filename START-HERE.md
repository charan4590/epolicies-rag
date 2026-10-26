# Start here — run ePolicies on your laptop with your own .md files

You need: your policy `.md` folder, a Google API key (with access to `gemini-embedding-001` and a
generation model such as `gemini-3.5-flash`), and about 30 minutes. Full details: [README.md](README.md).

## Windows quick start (no typing)

1. Install **Python 3.12** from python.org (tick **"Add python.exe to PATH"**).
2. Download this project: GitHub page -> green **Code** button -> **Download ZIP**.
3. Right-click the ZIP -> **Extract All...** -> type `C:\` as the destination -> **Extract**.
   You now have `C:\epolicies-rag-main` (with `backend`, `frontend`, `setup-windows.bat` inside).
4. Copy your policy folder (the one with the `.md` files) into `C:\epolicies-rag-main` and make sure it is
   named **`output`** -> `C:\epolicies-rag-main\output`.
5. Double-click **`setup-windows.bat`** in `C:\epolicies-rag-main`.
   It installs everything and checks all your files. Take a photo of the result.

The steps below explain the same thing in more detail (and the database / Gemini key parts).

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
git clone https://github.com/charan4590/epolicies-rag.git
cd epolicies-rag
```

(No Git? Download it as a ZIP: GitHub page → green **Code** button → **Download ZIP**.)

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

1. **The folder with your .md files** — use the folder that directly holds the policy `.md` files (for a scraper project like `C:\epolicies_project`, that is `C:\epolicies_project\output`, **not** the parent folder, otherwise `output_backup` and `history` copies get indexed too). Paste its full path (Windows: Shift+right-click the folder → *Copy as path*; Mac: Option+right-click → *Copy as Pathname*). It counts the files so you can confirm (~235). The files stay where they are and are never changed.
2. **How your company provides Gemini**: `1` = an API key (paste it; hidden while typing), or `2` = Vertex AI (Google Cloud project ID, region, and the path to the service-account `.json` file IT gives you). Then the answer model (default `gemini-3.5-flash`).
3. **Database details** (host, port, name, user, password).

It writes `epolicies-rag/.env`, creates the tables, tests everything, and offers to index your documents.

## 5. Check and index

Before you have the key or a database you can already check your files (offline, writes nothing):

```
python -m app.cli check-files     # read + chunk every .md file; lists broken, empty or duplicate files
```

```
python -m app.cli doctor          # every line should say [ OK ]
python -m app.cli ingest          # reads, chunks and embeds all .md files (safe to re-run)
python -m app.cli status          # how many documents/chunks are indexed
```

If `ingest` stops (quota, network), run `python -m app.cli resume` later — finished work is not repeated.

## 6. Start the app (two terminals)

**Windows, every day after setup:** double-click **`start-windows.bat`**. It starts your own PostgreSQL
database (`%USERPROFILE%\epolicies-pgdata`, port 5433), asks you to sign in again if the Google Cloud
(Vertex AI) login has expired, runs `doctor`, opens the backend and website in their own windows and
then opens http://localhost:3000. Double-click **`stop-windows.bat`** to stop everything. If PostgreSQL is
installed somewhere else, edit the three settings lines at the top of both files.

Manual alternative:

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

## Answers come only from your files

The assistant answers only from your indexed `.md` files and never searches the internet. If the
documents don't cover a question, it says so instead of guessing (`STRICT_GROUNDING=true`). The only
outside connection is the Google Gemini API (your company key). See "Your documents are the only
knowledge source" in README.md.

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

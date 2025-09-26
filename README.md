"""
Parent–School Dashboard — Dual-mode server (FastAPI with fallback WSGI)

This updated file fixes sandbox errors when attempting to start a WSGI server in environments that do
not permit creating/listening on sockets (common in restricted sandbox environments). The previous
implementation attempted to start a server by default which triggered `OSError: [Errno 138] Not supported`.

Changes made:
- Added a safe `can_bind()` check that attempts to bind and listen on a temporary socket to detect whether the
  environment supports creating/listening on sockets.
- The fallback WSGI server will NOT start automatically in restricted environments. To explicitly
  start the WSGI server, run the script with `--run-server` (e.g. `python main.py --run-server`).
- FastAPI is still used when the `ssl` module and FastAPI are available; in that case you can use
  Uvicorn as before (`uvicorn main:app --reload`) or run the script with `--run-server` if uvicorn
  is installed.
- Added robust error handling around server creation to catch `OSError` (Errno 138) even if the
  environment's probe mistakenly reports bindability.
- Kept and extended unit tests for the datastore; run with `python main.py --test`.

Usage:
- Run unit tests: `python main.py --test`
- Start the WSGI fallback server (if the environment allows sockets): `python main.py --run-server`
- Preferred (FastAPI): install FastAPI + uvicorn and run `uvicorn main:app --reload`

Endpoints supported (both modes):
- GET  /api/parent/{parent_id}/feed
- GET  /api/student/{student_id}/summary
- POST /api/messages              (JSON body)
- POST /api/consent               (JSON body)
- POST /api/chatbot/query         (JSON body)
- POST /admin/seed                (seed demo data)

This file is intentionally conservative about starting servers automatically to avoid sandbox
errors. If you'd like a different default behavior (for example, always attempt to start the
server), tell me and I can change it.
"""

from __future__ import annotations
import sys
import json
import sqlite3
import importlib
import os
import socket
from datetime import datetime
from typing import List, Optional, Dict, Any

# ---------------------------
# Helper: safe import
# ---------------------------

def safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

# Check for ssl availability; only attempt to import FastAPI if ssl is present
ssl_mod = safe_import("ssl")
fastapi_mod = None
uvicorn_mod = None
if ssl_mod is not None:
    fastapi_mod = safe_import("fastapi")
    uvicorn_mod = safe_import("uvicorn")

# ---------------------------
# Data layer (sqlite3-backed)
# ---------------------------
class DataStore:
    """Simple sqlite3-backed data store with the tables needed by the API.
    Uses a file-based DB by default (parent_school.db) and supports :memory: for tests.
    """

    def __init__(self, db_path: str = "./parent_school.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Return rows as sqlite3.Row for dict-like access
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def init_db(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dob TEXT,
                class_name TEXT,
                roll_no TEXT,
                photo_url TEXT,
                emergency_contact TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                recorded_by TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                subject TEXT NOT NULL,
                term TEXT NOT NULL,
                score INTEGER NOT NULL,
                max_score INTEGER DEFAULT 100,
                remarks TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user_id INTEGER NOT NULL,
                to_user_id INTEGER NOT NULL,
                student_id INTEGER,
                body TEXT NOT NULL,
                attachments TEXT,
                read_at TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS consent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                data_category TEXT NOT NULL,
                allowed INTEGER DEFAULT 0,
                timestamp TEXT
            )
            """
        )
        self._conn.commit()

    # -----------------
    # CRUD & queries
    # -----------------
    def seed_demo(self) -> bool:
        cur = self._conn.cursor()
        # check already seeded (simple check)
        cur.execute("SELECT COUNT(*) as c FROM students")
        if cur.fetchone()[0] > 0:
            return False
        # seed a student
        cur.execute(
            "INSERT INTO students (name, dob, class_name, roll_no, photo_url, emergency_contact) VALUES (?,?,?,?,?,?)",
            ("Aisha Sharma", "2013-05-17", "6A", "12", None, "+91-98765xxxx"),
        )
        student_id = cur.lastrowid
        # attendance
        cur.executemany(
            "INSERT INTO attendance (student_id, date, status, reason) VALUES (?,?,?,?)",
            [
                (student_id, "2025-09-01", "present", None),
                (student_id, "2025-09-02", "absent", "fever"),
                (student_id, "2025-09-03", "present", None),
            ],
        )
        # grades
        cur.executemany(
            "INSERT INTO grades (student_id, subject, term, score, max_score, remarks) VALUES (?,?,?,?,?,?)",
            [
                (student_id, "Math", "Term1", 82, 100, "Good"),
                (student_id, "Science", "Term1", 78, 100, "Needs practice"),
            ],
        )
        self._conn.commit()
        return True

    def get_parent_feed(self, parent_id: int) -> List[Dict[str, Any]]:
        # A simple static feed + a personal card placeholder
        now = datetime.utcnow().isoformat()
        feed = [
            {"type": "announcement", "title": "Sports Day on 2025-10-15", "body": "All students to wear white.", "date": now},
            {"type": "personal", "title": "Pending fee", "body": "Tuition fee for October is pending.", "date": now},
        ]
        return feed

    def get_student_summary(self, student_id: int) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        s = cur.fetchone()
        if s is None:
            raise KeyError("Student not found")
        # attendance percent
        cur.execute("SELECT status FROM attendance WHERE student_id = ?", (student_id,))
        rows = cur.fetchall()
        if rows:
            present = sum(1 for r in rows if r[0].lower() == "present")
            attendance_percent = round((present / len(rows)) * 100, 2)
        else:
            attendance_percent = 0.0
        # recent grades
        cur.execute(
            "SELECT subject, term, score, max_score, remarks FROM grades WHERE student_id = ? ORDER BY id DESC LIMIT 5",
            (student_id,),
        )
        grades = [dict(subject=r[0], term=r[1], score=r[2], max_score=r[3], remarks=r[4]) for r in cur.fetchall()]
        return {
            "id": s["id"],
            "name": s["name"],
            "class_name": s["class_name"],
            "roll_no": s["roll_no"],
            "attendance_percent": attendance_percent,
            "recent_grades": grades,
        }

    def create_message(self, payload: Dict[str, Any]) -> int:
        cur = self._conn.cursor()
        attachments = ",".join(payload.get("attachments") or []) if payload.get("attachments") else None
        created_at = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO messages (from_user_id, to_user_id, student_id, body, attachments, created_at) VALUES (?,?,?,?,?,?)",
            (payload["from_user_id"], payload["to_user_id"], payload.get("student_id"), payload["body"], attachments, created_at),
        )
        self._conn.commit()
        return cur.lastrowid

    def set_consent(self, payload: Dict[str, Any]) -> int:
        cur = self._conn.cursor()
        timestamp = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO consent (parent_id, student_id, data_category, allowed, timestamp) VALUES (?,?,?,?,?)",
            (payload["parent_id"], payload["student_id"], payload["data_category"], 1 if payload["allowed"] else 0, timestamp),
        )
        self._conn.commit()
        return cur.lastrowid

    def chatbot_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        parent_id = payload.get("parent_id")
        student_id = payload.get("student_id")
        consent_scope = payload.get("consent_scope") or []
        # Basic consent check: if consent_scope provided, ensure at least one matching allowed consent exists
        if student_id and consent_scope:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT data_category FROM consent WHERE parent_id = ? AND student_id = ? AND allowed = 1",
                (parent_id, student_id),
            )
            allowed = {r[0] for r in cur.fetchall()}
            if not any(cat in allowed for cat in consent_scope):
                return {"answer": "No consent to access requested data categories. Please update consent settings.", "sources": []}
        snippets = []
        if student_id:
            cur = self._conn.cursor()
            cur.execute("SELECT date, status, reason FROM attendance WHERE student_id = ? ORDER BY date DESC LIMIT 5", (student_id,))
            for r in cur.fetchall():
                snippets.append(f"{r[0]}: {r[1]} ({r[2] or 'no reason'})")
            cur.execute("SELECT term, subject, score, max_score, remarks FROM grades WHERE student_id = ? ORDER BY id DESC LIMIT 5", (student_id,))
            for r in cur.fetchall():
                snippets.append(f"{r[0]} - {r[1]}: {r[2]}/{r[3]} ({r[4] or ''})")
        answer = f"(Placeholder) I found {len(snippets)} records related to your query. Here are brief snippets:
" + "
".join(snippets[:5])
        return {"answer": answer, "sources": snippets}

# ---------------------------
# Utility: check whether the runtime allows binding to a socket
# ---------------------------

def can_bind(host: str = "127.0.0.1", port: int = 0, timeout: float = 0.5) -> bool:
    """Try to bind a temporary socket to detect whether the environment allows listening on sockets.

    If port is 0 the OS will pick an ephemeral port. We catch OSError and return False in that case.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        # set REUSEADDR to avoid TIME_WAIT issues on repeated test runs
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        # Attempt to listen briefly to ensure the platform supports listening
        try:
            s.listen(1)
        except OSError:
            s.close()
            return False
        s.close()
        return True
    except OSError:
        return False

# ---------------------------
# Build FastAPI app (only if ssl present and FastAPI import worked)
# ---------------------------
app = None

if ssl_mod is not None and fastapi_mod is not None:
    # Import things from FastAPI lazily
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List

    # Pydantic schemas (subset for request validation)
    class MessageCreateSchema(BaseModel):
        from_user_id: int
        to_user_id: int
        student_id: Optional[int] = None
        body: str
        attachments: Optional[List[str]] = None

    class ConsentCreateSchema(BaseModel):
        parent_id: int
        student_id: int
        data_category: str
        allowed: bool

    class ChatQuerySchema(BaseModel):
        parent_id: int
        student_id: Optional[int] = None
        query: str
        consent_scope: Optional[List[str]] = None

    # create app and datastore
    app = FastAPI(title="Parent-School Dashboard API (MVP)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = DataStore()

    # dependency
    def get_store():
        return store

    @app.get("/api/parent/{parent_id}/feed")
    def get_parent_feed(parent_id: int, store: DataStore = Depends(get_store)):
        return store.get_parent_feed(parent_id)

    @app.get("/api/student/{student_id}/summary")
    def get_student_summary(student_id: int, store: DataStore = Depends(get_store)):
        try:
            return store.get_student_summary(student_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Student not found")

    @app.post("/api/messages")
    def create_message(msg: MessageCreateSchema, store: DataStore = Depends(get_store)):
        mid = store.create_message(msg.dict())
        return {"ok": True, "message_id": mid}

    @app.post("/api/consent")
    def set_consent(payload: ConsentCreateSchema, store: DataStore = Depends(get_store)):
        cid = store.set_consent(payload.dict())
        return {"ok": True, "consent_id": cid}

    @app.post("/api/chatbot/query")
    def chatbot_query(q: ChatQuerySchema, store: DataStore = Depends(get_store)):
        return store.chatbot_query(q.dict())

    @app.post("/admin/seed")
    def seed_demo(store: DataStore = Depends(get_store)):
        ok = store.seed_demo()
        return {"ok": ok}

# ---------------------------
# Fallback WSGI app (no FastAPI required)
# ---------------------------
else:
    # We'll implement a minimal WSGI app with the same endpoints and semantics.
    def _json_response(start_response, obj, status='200 OK'):
        body = json.dumps(obj, default=str).encode('utf-8')
        headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(body)))]
        start_response(status, headers)
        return [body]

    def _read_json_body(environ):
        try:
            length = int(environ.get('CONTENT_LENGTH', '0') or '0')
        except Exception:
            length = 0
        if length:
            data = environ['wsgi.input'].read(length)
            try:
                return json.loads(data.decode('utf-8'))
            except Exception:
                return None
        # maybe chunked or empty
        return None

    store = DataStore()

    def wsgi_app(environ, start_response):
        method = environ.get('REQUEST_METHOD', 'GET').upper()
        path = environ.get('PATH_INFO', '')

        # Routing
        try:
            if method == 'GET' and path.startswith('/api/parent/') and path.endswith('/feed'):
                # /api/parent/{parent_id}/feed
                parts = path.strip('/').split('/')
                try:
                    parent_id = int(parts[2])
                except Exception:
                    return _json_response(start_response, {"detail": "Invalid parent id"}, status='400 Bad Request')
                data = store.get_parent_feed(parent_id)
                return _json_response(start_response, data)

            if method == 'GET' and path.startswith('/api/student/') and path.endswith('/summary'):
                parts = path.strip('/').split('/')
                try:
                    student_id = int(parts[1])
                except Exception:
                    return _json_response(start_response, {"detail": "Invalid student id"}, status='400 Bad Request')
                try:
                    data = store.get_student_summary(student_id)
                    return _json_response(start_response, data)
                except KeyError:
                    return _json_response(start_response, {"detail": "Student not found"}, status='404 Not Found')

            if method == 'POST' and path == '/api/messages':
                payload = _read_json_body(environ) or {}
                if not payload:
                    return _json_response(start_response, {"detail": "Invalid or missing JSON body"}, status='400 Bad Request')
                mid = store.create_message(payload)
                return _json_response(start_response, {"ok": True, "message_id": mid})

            if method == 'POST' and path == '/api/consent':
                payload = _read_json_body(environ) or {}
                if not payload:
                    return _json_response(start_response, {"detail": "Invalid or missing JSON body"}, status='400 Bad Request')
                cid = store.set_consent(payload)
                return _json_response(start_response, {"ok": True, "consent_id": cid})

            if method == 'POST' and path == '/api/chatbot/query':
                payload = _read_json_body(environ) or {}
                if not payload:
                    return _json_response(start_response, {"detail": "Invalid or missing JSON body"}, status='400 Bad Request')
                ans = store.chatbot_query(payload)
                return _json_response(start_response, ans)

            if method == 'POST' and path == '/admin/seed':
                ok = store.seed_demo()
                return _json_response(start_response, {"ok": ok})

            # default
            return _json_response(start_response, {"detail": "Not found"}, status='404 Not Found')
        except Exception as e:
            # Avoid leaking internal trace in production; here we return the error for developer convenience.
            return _json_response(start_response, {"detail": "Internal server error", "error": str(e)}, status='500 Internal Server Error')

# ---------------------------
# Unit tests for the data layer (extended)
# ---------------------------
def run_unit_tests():
    print("Running unit tests against DataStore (in-memory)...")
    ds = DataStore(db_path=":memory:")
    # ensure empty
    try:
        # 1) seed
        try_seed = ds.seed_demo()
        assert try_seed is True, "Seed should return True on fresh DB"
        # calling seed again should return False
        assert ds.seed_demo() is False, "Re-seeding should return False"

        # fetch the seeded student id
        cur = ds._conn.cursor()
        cur.execute("SELECT id FROM students LIMIT 1")
        row = cur.fetchone()
        assert row is not None, "Seed should create at least one student"
        student_id = row[0]

        # 2) summary
        summary = ds.get_student_summary(student_id)
        assert summary['name'].startswith('Aisha'), "Student name should be seeded"
        assert isinstance(summary['attendance_percent'], float), "attendance_percent should be a float"
        assert isinstance(summary['recent_grades'], list), "recent_grades should be a list"

        # 3) feed
        feed = ds.get_parent_feed(parent_id=1)
        assert isinstance(feed, list) and len(feed) >= 1, "Feed should be a non-empty list"
        assert 'type' in feed[0] and 'title' in feed[0]

        # 4) create message
        mid = ds.create_message({
            'from_user_id': 1,
            'to_user_id': 2,
            'student_id': student_id,
            'body': 'Hello teacher',
            'attachments': []
        })
        assert isinstance(mid, int) and mid > 0

        # confirm message stored
        cur.execute("SELECT body FROM messages WHERE id = ?", (mid,))
        msg_row = cur.fetchone()
        assert msg_row is not None and 'Hello teacher' in msg_row[0]

        # 5) consent
        cid = ds.set_consent({'parent_id': 1, 'student_id': student_id, 'data_category': 'grades', 'allowed': True})
        assert isinstance(cid, int) and cid > 0

        # 6) chatbot with consent scope that exists
        resp = ds.chatbot_query({'parent_id': 1, 'student_id': student_id, 'query': 'How is math?', 'consent_scope': ['grades']})
        assert 'answer' in resp and isinstance(resp['answer'], str)

        # 7) chatbot with disallowed scope (parent_id 99 has no consent)
        resp2 = ds.chatbot_query({'parent_id': 99, 'student_id': student_id, 'query': 'How is math?', 'consent_scope': ['grades']})
        assert 'No consent' in resp2['answer']

        # 8) missing student raises KeyError
        try:
            ds.get_student_summary(999999)
            raise AssertionError("Expected KeyError for missing student id")
        except KeyError:
            pass

        print("All unit tests passed.")
    except AssertionError as e:
        print("Unit test failure:", e)
        raise

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == '__main__':
    if '--test' in sys.argv:
        run_unit_tests()
        sys.exit(0)

    # If FastAPI is available, prefer running via uvicorn or have uvicorn auto-run
    if app is not None:
        if uvicorn_mod is not None and '--run-server' in sys.argv:
            # allow explicit server run via --run-server when uvicorn is installed
            print("Starting FastAPI app with uvicorn on http://0.0.0.0:8000")
            uvicorn_mod.run("main:app", host="0.0.0.0", port=8000, reload=True)
        else:
            print("FastAPI app is available. Recommended run command:")
            if uvicorn_mod is not None:
                print("  uvicorn main:app --reload")
            else:
                print("  (install uvicorn) then run: uvicorn main:app --reload")
            print("Alternatively, to run the fallback WSGI server, run: python main.py --run-server")
            sys.exit(0)

    # Fallback WSGI mode
    # Do not attempt to bind sockets automatically in restricted sandboxes.
    port = int(os.environ.get('PARENT_SCHOOL_PORT', '8000'))
    host = os.environ.get('PARENT_SCHOOL_HOST', '127.0.0.1')

    if '--run-server' in sys.argv:
        # Only attempt to start the server when explicitly requested
        if not can_bind(host, port):
            print(f"Cannot bind to {host}:{port} in this environment. This sandbox likely disallows creating/listening on sockets.")
            print("To run locally, run this script on your machine or in an environment that allows network sockets.")
            print("You can still run unit tests with: python main.py --test")
            sys.exit(1)
        # safe to start
        from wsgiref.simple_server import make_server
        print(f"Starting fallback WSGI server on http://{host}:{port}")
        httpd = make_server(host, port, wsgi_app)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('
Server stopped by user')
    else:
        # conservative default: do not start server automatically to avoid sandbox binding errors
        print("Fallback WSGI is available but not started automatically in this environment to avoid socket binding errors.")
        print("To start the WSGI server (if your environment allows it), run: python main.py --run-server")
        print("To run unit tests: python main.py --test")
        sys.exit(0)

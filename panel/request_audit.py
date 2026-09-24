"""Pure ASGI vLLM request audit middleware. No request/response mutation."""
from __future__ import annotations
import asyncio
from contextlib import contextmanager
import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from pathlib import Path

LOG = logging.getLogger(__name__)
LIMIT = int(os.environ.get('AUDIT_BODY_LIMIT', '2097152'))

SCHEMA = '''CREATE TABLE IF NOT EXISTS requests (
 id TEXT PRIMARY KEY, started REAL, endpoint TEXT, model TEXT, task TEXT,
 status INTEGER, outcome TEXT, duration_ms REAL, ttft_ms REAL,
 input_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
 reasoning_tokens INTEGER, request_body TEXT, response_body TEXT,
 truncated INTEGER, usage_json TEXT, metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS requests_started ON requests(started);'''

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=32)
        self.dropped = 0
        self.written = 0
        self.last_error = None
        with self.connect() as db:
            db.executescript(SCHEMA)
        self.path.chmod(0o640)
        threading.Thread(target=self.run, daemon=True).start()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute('PRAGMA journal_mode=WAL')
            with db:
                yield db
        finally:
            db.close()

    def submit(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            LOG.error('Request audit queue full; record dropped')

    def run(self):
        last_cleanup = 0
        # Keep one writer connection open so read-only panel clients can use
        # the existing WAL/SHM files without needing directory write access.
        db = sqlite3.connect(self.path, timeout=5)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('SELECT COUNT(*) FROM requests').fetchone()
        while True:
            row = self.queue.get()
            try:
                with db:
                    db.execute('INSERT OR REPLACE INTO requests VALUES (' + ','.join('?' for _ in row) + ')', row)
                    if time.time() - last_cleanup > 3600:
                        days = max(1, int(os.environ.get('AUDIT_RETENTION_DAYS', '7')))
                        db.execute('DELETE FROM requests WHERE started < ?', (time.time()-days*86400,))
                        last_cleanup = time.time()
                self.written += 1
                self.last_error = None
            except Exception as exc:
                self.dropped += 1
                self.last_error = type(exc).__name__
                LOG.exception('Request audit write failed')
            finally:
                self.queue.task_done()

class Capture:
    def __init__(self):
        self.body = bytearray()
        self.truncated = False
        self.pending = bytearray()
        self.discard_event = False
        self.usage = None
        self.metrics = None
        self.first_token = None
        self.request_id = None
        self.terminal = False
        self.failed = False

    def event(self, obj):
        if not isinstance(obj, dict):
            return
        root = obj.get('response') if isinstance(obj.get('response'), dict) else obj
        if isinstance(root.get('usage'), dict):
            self.usage = root['usage']
        if isinstance(root.get('metrics'), dict):
            self.metrics = root['metrics']
        if root.get('id'):
            self.request_id = root['id']
        kind = obj.get('type', '')
        if obj.get('error') or kind in ('response.failed', 'response.incomplete', 'error'):
            self.failed = True
        if kind in ('response.completed', 'response.failed', 'response.incomplete'):
            self.terminal = True
        choices = obj.get('choices') or []
        has_token = any(isinstance(c, dict) and (c.get('text') or any((c.get('delta') or {}).get(k) for k in ('content','reasoning','reasoning_content','tool_calls'))) for c in choices)
        has_token = has_token or (kind.endswith('.delta') and bool(obj.get('delta')))
        if has_token and self.first_token is None:
            self.first_token = time.monotonic()

    def feed(self, data, streaming):
        room = max(0, LIMIT-len(self.body))
        self.body.extend(data[:room])
        self.truncated |= len(data) > room
        if not streaming:
            return
        # Process SSE lines incrementally even after the stored body reaches its cap.
        for segment in data.splitlines(keepends=True):
            self.pending.extend(segment)
            if len(self.pending) > LIMIT:
                self.pending.clear()
                self.discard_event = True
            if segment.endswith(b'\n'):
                line = bytes(self.pending).strip()
                self.pending.clear()
                if not self.discard_event and line.startswith(b'data:'):
                    value = line[5:].strip()
                    if value == b'[DONE]':
                        self.terminal = True
                    else:
                        try:
                            self.event(json.loads(value))
                        except (ValueError, UnicodeError):
                            pass
                self.discard_event = False

class RequestAuditMiddleware:
    def __init__(self, app):
        self.app = app
        self.store = Store(os.environ.get('AUDIT_DB', '/audit/requests.db'))

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            async def lifecycle_send(message):
                if message['type'] == 'lifespan.shutdown.complete':
                    await asyncio.to_thread(self.store.queue.join)
                await send(message)
            return await self.app(scope, receive, lifecycle_send)
        if scope['type'] == 'http' and scope['path'] == '/audit-health':
            body = json.dumps({'written': self.store.written, 'dropped': self.store.dropped, 'queued': self.store.queue.qsize(), 'error': self.store.last_error}).encode()
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':body})
            return
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope['path'] not in ('/v1/chat/completions','/v1/completions','/v1/responses'):
            return await self.app(scope, receive, send)
        started, tick = time.time(), time.monotonic()
        request = bytearray()
        truncated = False
        response = Capture()
        status, streaming, complete = 500, False, False
        outcome = 'incomplete'
        async def read():
            nonlocal truncated, outcome
            msg = await receive()
            if msg['type'] == 'http.request':
                data = msg.get('body', b'')
                room = max(0, LIMIT-len(request))
                request.extend(data[:room])
                truncated |= len(data) > room
            if msg['type'] == 'http.disconnect':
                outcome = 'disconnected'
            return msg
        async def write(msg):
            nonlocal status, streaming, complete
            if msg['type'] == 'http.response.start':
                status = msg['status']
                streaming = any(k.lower() == b'content-type' and b'text/event-stream' in v for k,v in msg.get('headers', []))
            if msg['type'] == 'http.response.body':
                response.feed(msg.get('body',b''), streaming)
            await send(msg)
            if msg['type'] == 'http.response.body' and not msg.get('more_body',False):
                complete = True
        try:
            await self.app(scope, read, write)
            if complete:
                outcome = ('ok' if status < 400 and not response.failed and (not streaming or response.terminal) else 'error' if status >= 400 or response.failed else 'incomplete')
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        except Exception:
            outcome = 'error'
            raise
        finally:
            # Observability must never turn a successful inference into an error.
            try:
                if not streaming and not response.truncated:
                    try:
                        response.event(json.loads(response.body))
                    except (ValueError, UnicodeError):
                        pass
                try:
                    payload = json.loads(request)
                    if not isinstance(payload, dict): payload = {}
                except (ValueError, UnicodeError):
                    payload = {}
                headers = dict(scope.get('headers',[]))
                task = headers.get(b'x-task-id', b'').decode(errors='replace')[:200] or None
                usage = response.usage or {}
                details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
                output_details = usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}
                metrics = response.metrics or {}
                metrics['upstream_request_id'] = response.request_id
                metrics['caller_id'] = headers.get(b'x-audit-caller', b'').decode(errors='replace')[:200] or None
                ttft = metrics.get('time_to_first_token_ms')
                if ttft is None and response.first_token:
                    ttft = (response.first_token-tick)*1000
                self.store.submit((
                    uuid.uuid4().hex, started, scope['path'], str(payload.get('model',''))[:200], task,
                    status, outcome, round((time.monotonic()-tick)*1000,3), ttft,
                    usage.get('prompt_tokens',usage.get('input_tokens')), details.get('cached_tokens'),
                    usage.get('completion_tokens',usage.get('output_tokens')), output_details.get('reasoning_tokens'),
                    request.decode(errors='replace'), response.body.decode(errors='replace'),
                    int(truncated or response.truncated), json.dumps(usage), json.dumps(metrics)
                ))
            except Exception:
                LOG.exception('Request audit capture failed')

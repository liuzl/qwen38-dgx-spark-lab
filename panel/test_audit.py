import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from panel import request_audit as audit
from panel.audit_reader import AuditReader

class AuditTests(unittest.TestCase):
    def test_split_sse_after_truncation(self):
        capture=audit.Capture()
        usage={'prompt_tokens':100,'completion_tokens':2,'prompt_tokens_details':{'cached_tokens':80}}
        body=b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        body+=('data: '+json.dumps({'choices':[],'usage':usage})+'\n\ndata: [DONE]\n\n').encode()
        with patch.object(audit,'LIMIT',160):
            for offset in range(0,len(body),7): capture.feed(body[offset:offset+7],True)
        self.assertTrue(capture.truncated)
        self.assertEqual(capture.usage,usage)
        self.assertTrue(capture.terminal)
        self.assertIsNotNone(capture.first_token)

    def test_middleware_transparency_and_reader(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ',{'AUDIT_DB':folder+'/audit.db'}):
            response={'id':'upstream-1','usage':{'prompt_tokens':10,'completion_tokens':3,'prompt_tokens_details':{'cached_tokens':8}}}
            async def app(scope,receive,send):
                incoming=await receive()
                self.assertEqual(incoming['body'],b'{"model":"test"}')
                await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]})
                await send({'type':'http.response.body','body':json.dumps(response).encode()})
            middleware=audit.RequestAuditMiddleware(app)
            messages=[]
            async def receive(): return {'type':'http.request','body':b'{"model":"test"}'}
            async def send(msg): messages.append(msg)
            asyncio.run(middleware({'type':'http','method':'POST','path':'/v1/chat/completions','headers':[(b'authorization',b'secret'),(b'x-task-id',b'test-task')]},receive,send))
            middleware.store.queue.join()
            self.assertEqual(json.loads(messages[1]['body']),response)
            reader=AuditReader(folder+'/audit.db')
            summary=reader.summary(1)
            self.assertEqual(summary['requests'],1)
            self.assertEqual(summary['cached_tokens'],8)
            row=reader.requests(1,task='test-task')['requests'][0]
            self.assertNotIn('request_body',row)
            self.assertIsNone(row['ttft_ms'])
            self.assertEqual(row['outcome'],'ok')
            self.assertNotIn('secret',json.dumps(reader.detail(row['id'])))
            self.assertEqual(reader.requests(1,task="' OR 1=1 --")['requests'],[])

    def test_responses_usage(self):
        capture=audit.Capture()
        capture.feed(b'data: {"type":"response.completed","response":{"id":"resp-1","usage":{"input_tokens":12,"output_tokens":3}}}\n\n',True)
        self.assertEqual(capture.usage['input_tokens'],12)
        self.assertTrue(capture.terminal)

    def test_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ',{'AUDIT_DB':folder+'/audit.db'}):
            async def app(scope,receive,send): raise RuntimeError('test')
            middleware=audit.RequestAuditMiddleware(app)
            async def noop(*args): pass
            with self.assertRaises(RuntimeError):
                asyncio.run(middleware({'type':'http','method':'POST','path':'/v1/responses'},noop,noop))
            middleware.store.queue.join()
            row=AuditReader(folder+'/audit.db').requests(1)['requests'][0]
            self.assertEqual(row['outcome'],'error')
            self.assertIsNone(row['input_tokens'])

    def test_exclude_smoke_and_latest_outside_window(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audit.db'
            now = time.time()
            with sqlite3.connect(path) as db:
                db.executescript(audit.SCHEMA)
                for identifier, started, task in [('normal', now-7200, None),
                        ('smoke', now, 'audit-cache-smoke-20260925'),
                        ('generic', now-7100, 'audit-production-review')]:
                    db.execute('INSERT INTO requests (id, started, task, outcome, duration_ms, metrics_json) VALUES (?, ?, ?, ?, ?, ?)',
                        (identifier, started, task, 'ok', 200, '{}'))
            reader = AuditReader(path)
            self.assertEqual(reader.summary(1)['requests'], 1)
            self.assertEqual(reader.summary(1, True)['requests'], 0)
            self.assertEqual(reader.summary(1, True)['latest_request'], now-7100)
            rows = reader.requests(24, exclude_tests=True)['requests']
            self.assertEqual({row['id'] for row in rows}, {'normal', 'generic'})

    def test_audit_endpoints_without_extra_key(self):
        import threading
        import sqlite3
        import urllib.request
        import urllib.error
        from http.server import ThreadingHTTPServer
        from panel import server
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ',{'AUDIT_DB':folder+'/audit.db'}):
            store=audit.Store(folder+'/audit.db')
            with patch.object(server,'AUDIT_READER',AuditReader(folder+'/audit.db')):
                http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
                threading.Thread(target=http.serve_forever,daemon=True).start()
                url=f'http://127.0.0.1:{http.server_port}/api/audit/'
                try:
                    with urllib.request.urlopen(url+'summary') as r: self.assertEqual(r.status,200)
                    with urllib.request.urlopen(url+'requests') as r:
                        self.assertEqual(json.load(r)['requests'], [])
                    with sqlite3.connect(folder+'/audit.db') as db:
                        db.execute('INSERT INTO requests (id, started, outcome, request_body, response_body, metrics_json, usage_json) VALUES (?, ?, ?, ?, ?, ?, ?)',
                            ('fixture', time.time(), 'ok', '{"prompt":"hello"}', '{"text":"world"}', '{}', '{}'))
                    with urllib.request.urlopen(url+'requests/fixture') as r:
                        detail = json.load(r)
                        self.assertEqual(detail['request_body'], '{"prompt":"hello"}')
                        self.assertEqual(detail['response_body'], '{"text":"world"}')
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(url+'requests/missing')
                    self.assertEqual(error.exception.code, 404)
                finally:
                    http.shutdown();http.server_close()

if __name__=='__main__': unittest.main()

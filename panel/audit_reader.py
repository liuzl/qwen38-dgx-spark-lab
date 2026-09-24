"""Read-only access to the audit database; bodies require separate authorization."""
import json
from contextlib import contextmanager
import math
import sqlite3
import time
from pathlib import Path

class AuditReader:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        try:
            yield db
        finally:
            db.close()

    def summary(self, hours):
        cutoff = time.time()-hours*3600
        with self.connect() as db:
            row = dict(db.execute('''SELECT COUNT(*) AS requests,
                SUM(CASE WHEN outcome != 'ok' THEN 1 ELSE 0 END) AS errors,
                SUM(input_tokens) AS input_tokens, SUM(cached_tokens) AS cached_tokens,
                SUM(output_tokens) AS output_tokens, SUM(reasoning_tokens) AS reasoning_tokens,
                SUM(CASE WHEN input_tokens IS NULL THEN 1 ELSE 0 END) AS missing_usage,
                SUM(truncated) AS truncated, AVG(duration_ms) AS duration_avg_ms,
                AVG(ttft_ms) AS ttft_avg_ms,
                AVG(json_extract(metrics_json, '$.tokens_per_second')) AS decode_tok_s,
                AVG(json_extract(metrics_json, '$.queue_time_ms')) AS queue_avg_ms, MIN(started) AS first_request,
                MAX(started) AS last_request FROM requests WHERE started >= ?''', (cutoff,)).fetchone())
            for column, prefix in [('duration_ms','duration'),('ttft_ms','ttft')]:
                count = db.execute(f'SELECT COUNT({column}) FROM requests WHERE started>=?', (cutoff,)).fetchone()[0]
                for q, name in [(0.5,'p50'),(0.95,'p95')]:
                    value = db.execute(f'SELECT {column} FROM requests WHERE started>=? AND {column} IS NOT NULL ORDER BY {column} LIMIT 1 OFFSET ?', (cutoff,max(0,math.ceil(count*q)-1))).fetchone()
                    row[f'{prefix}_{name}_ms'] = value[0] if value else None
            row['retained_since'] = db.execute('SELECT MIN(started) FROM requests').fetchone()[0]
        row['hours'] = hours
        row['enabled'] = True
        return row

    def requests(self, hours, before=None, task='', model=''):
        where = ['started >= ?']
        args = [time.time()-hours*3600]
        if before is not None:
            where.append('started < ?'); args.append(before)
        if task:
            where.append('task = ?'); args.append(task)
        if model:
            where.append('model = ?'); args.append(model)
        with self.connect() as db:
            rows = db.execute('''SELECT id,started,endpoint,model,task,status,outcome,duration_ms,
                ttft_ms,input_tokens,cached_tokens,output_tokens,reasoning_tokens,truncated,
                json_extract(metrics_json, '$.caller_id') AS caller_id
                FROM requests WHERE ''' + ' AND '.join(where) + ' ORDER BY started DESC LIMIT 50', args).fetchall()
        items = [dict(row) for row in rows]
        return {'requests':items,'next_before':items[-1]['started'] if len(items)==50 else None}

    def detail(self, request_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone()
        if row is None: return None
        item = dict(row)
        for key in ('usage_json','metrics_json'):
            item[key] = json.loads(item[key])
        return item

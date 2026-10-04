#!/usr/bin/env python3
"""Synthetic multi-image qualification. No credentials or request/response text in results.

Fixtures: 01.png..16.png, each 1024x768, visibly labelled CODE 7400..7415.
Protocol and capacity runs are separate. Capacity uses streamed Chat, fresh cache_salt
per repetition/client, followed by an exact replay (warm); actual token/cache counts
are recorded. Run against a drained node, one node at a time.
"""
import argparse, base64, concurrent.futures, json, re, time, urllib.error, urllib.request, uuid
from pathlib import Path


def request(base, path, body):
    t0 = time.monotonic(); first = None; text = ''; usage = None; status = None
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'tf-media-qualification/1', 'anthropic-version': '2023-06-01'})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            status = r.status
            if body.get('stream'):
                for line in r:
                    if not line.startswith(b'data:'): continue
                    raw = line[5:].strip()
                    if raw == b'[DONE]': continue
                    e = json.loads(raw)
                    if e.get('usage'): usage = e['usage']
                    for choice in e.get('choices', []):
                        delta = choice.get('delta', {}).get('content') or ''
                        if delta and first is None: first = time.monotonic() - t0
                        text += delta
            else:
                e = json.load(r); usage = e.get('usage')
                if path.endswith('/messages'): text = ''.join(x.get('text', '') for x in e.get('content', []))
                elif path.endswith('/responses'): text = ''.join(c.get('text', '') for x in e.get('output', []) for c in x.get('content', []))
                else: text = e['choices'][0]['message'].get('content') or ''
    except urllib.error.HTTPError as e: status = e.code
    except Exception as e: return {'status': status, 'error': type(e).__name__, 'seconds': round(time.monotonic()-t0, 3)}, ''
    return {'status': status, 'seconds': round(time.monotonic()-t0, 3), 'ttft_s': round(first, 3) if first is not None else None, 'usage': usage}, text


def payload(model, images, protocol='chat', context_words=0, salt=None):
    question = 'Read CODE in the LAST image only. Reply with its four digits only.' if images else 'Reply exactly: text-ok'
    prefix = ('This is synthetic background context for a browser screenshot test. ' * context_words)
    body = {'model': model, 'temperature': 0, 'cache_salt': salt or str(uuid.uuid4())}
    if protocol == 'responses':
        body.update(max_output_tokens=32, reasoning={'effort':'none'}, input=[{'role':'user','content':[{'type':'input_text','text':prefix + question}] + [{'type':'input_image','detail':'auto','image_url':'data:image/png;base64,'+x} for x in images]}])
        path = '/v1/responses'
    elif protocol == 'messages':
        parts = [{'type':'image','source':{'type':'base64','media_type':'image/png','data':x}} for x in images]
        half = len(parts)//2
        body.update(max_tokens=32, messages=[{'role':'user','content':prefix + 'Inspect two batches of screenshots.'},
            {'role':'assistant','content':[{'type':'tool_use','id':'shot1','name':'browser','input':{}}]},
            {'role':'user','content':[{'type':'tool_result','tool_use_id':'shot1','content':parts[:half]}]},
            {'role':'assistant','content':[{'type':'tool_use','id':'shot2','name':'browser','input':{}}]},
            {'role':'user','content':[{'type':'tool_result','tool_use_id':'shot2','content':parts[half:]},{'type':'text','text':question}]}])
        path = '/v1/messages'
    else:
        body.update(max_tokens=32, stream=True, stream_options={'include_usage':True}, messages=[{'role':'user','content':[{'type':'text','text':prefix + question}] + [{'type':'image_url','image_url':{'url':'data:image/png;base64,'+x}} for x in images]}])
        path = '/v1/chat/completions'
    return path, body


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',required=True);p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--mode',choices=['protocols','capacity','quick'],default='protocols');p.add_argument('--repetitions',type=int,default=2)
    args=p.parse_args();images=[base64.b64encode((args.fixtures/f'{i:02d}.png').read_bytes()).decode() for i in range(1,17)]
    rows=[]
    def save():
        args.output.write_text(json.dumps({'label':args.label,'fixture':'16 distinct 1024x768 PNG synthetic browser pages; CODE 7400..7415','mode':args.mode,'repetitions':args.repetitions if args.mode=='capacity' else 1,'rows':rows,'all_passed':all(r.get('ok') for r in rows)},indent=2)+'\n')
    def run(model,n,protocol,context_words=0,c=1,rep=0):
        paths=[payload(model,images[:n],protocol,context_words) for _ in range(c)]
        phases=['cold','warm'] if args.mode=='capacity' else ['fresh']
        for phase in phases:
            t=time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c+1) as pool:
                jobs=[pool.submit(request,args.base_url,path,body) for path,body in paths]
                sentinel = pool.submit(request,args.base_url,*payload(model,[])) if args.mode=='capacity' and n else None
                results=[j.result() for j in jobs]
                text_result=sentinel.result() if sentinel else None
            for result,text in results:
                result.update(ok=result['status']==200 and (re.findall(r'\b\d{4}\b',text)==[str(7400+n-1)] if n else 'text-ok' in text), model=model,images=n,protocol=protocol,context_repetitions=context_words,concurrency=c,cache_phase=phase,repetition=rep)
                rows.append(result)
            if text_result:
                r,text=text_result;r.update(ok=r['status']==200 and 'text-ok' in text,model=model,images=0,protocol='chat',competing_images=n,context_repetitions=context_words,concurrency=c,cache_phase=phase,repetition=rep);rows.append(r)
            save();ok=all(r['ok'] for r,_ in results) and (text_result is None or rows[-1]['ok'])
            print(json.dumps({'model':model,'images':n,'protocol':protocol,'context_repetitions':context_words,'c':c,'phase':phase,'rep':rep,'ok':ok,'wall_s':round(time.monotonic()-t,2)}),flush=True)
            if not ok: raise SystemExit('qualification failed; inspect metadata output')
    models=['qwen3.8-27b','qwen3.8-27b-uncensored']
    if args.mode in ['protocols','quick']:
        for model in (models if args.mode=='protocols' else models[:1]):
            for n in ([6,8,16] if args.mode=='protocols' else [16]):
                for protocol in (['chat','responses','messages'] if args.mode=='protocols' else ['chat']): run(model,n,protocol)
        for protocol in ['chat','responses','messages']:
            path,body=payload(models[0],images+[images[0]],protocol)
            r,_=request(args.base_url,path,body);r.update(ok=r['status']==400,images=17,protocol=protocol,guardrail=True);rows.append(r);save()
        if not all(r['ok'] for r in rows): raise SystemExit('17-image guardrail failed')
    else:
        for model in models:
            for rep in range(args.repetitions):
                run(model,0,'chat',rep=rep)
                for context,c in [(0,1),(0,4),(0,8),(1500,1),(1500,4)]:run(model,16,'chat',context,c,rep)
    print('PASS',len(rows),'requests',flush=True)

if __name__=='__main__':main()

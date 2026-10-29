import json,time,pathlib,urllib.request,concurrent.futures,sys,os
os.umask(0o077)
root=pathlib.Path('/root/lfm3-private');variant=sys.argv[1];out=root/variant;out.mkdir(exist_ok=True)
samples={p.stem:json.loads(p.read_text()) for p in (root/'samples').glob('*.json') if p.stem in ['short1','short2','medium2','long1'] or p.stem.startswith('synthetic_')}
base='http://127.0.0.1:30000'
def flush():
 req=urllib.request.Request(base+'/flush_cache',data=b'{}',headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r:r.read()
def run(i,phase,rep=0,c=1,thinking=False,profile='greedy'):
 s=samples[i];payload={'model':'lfm3','messages':s['messages'],'temperature':0 if profile=='greedy' else (0.1 if profile=='official' else s.get('temperature',.8)),'top_p':1.0,'top_k':-1,'repetition_penalty':1.0,'max_tokens':512,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':thinking}}
 if profile=='official':payload.update(top_k=50,repetition_penalty=1.1)
 rec=dict(variant=variant,id=i,phase=phase,repeat=rep,concurrency=c,thinking=thinking,profile=profile,temperature=payload['temperature'])
 t=time.perf_counter();first=None;visible=None;content='';reason='';usage={};finish=None;chunks=0
 try:
  req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as r:
   for line in r:
    if not line.startswith(b'data: '):continue
    try:d=json.loads(line[6:])
    except ValueError:continue
    if d.get('usage'):usage=d['usage']
    for ch in d.get('choices',[]):
     delta=ch.get('delta',{});v=delta.get('content') or '';a=delta.get('reasoning_content') or delta.get('reasoning') or ''
     if (v or a) and first is None:first=time.perf_counter()-t
     if v and visible is None:visible=time.perf_counter()-t
     content+=v;reason+=a;chunks+=bool(v or a)
     if ch.get('finish_reason'):finish=ch['finish_reason']
  rec.update(duration_s=time.perf_counter()-t,ttft_any_s=first,ttft_visible_s=visible,finish=finish,input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'),cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',usage.get('reasoning_tokens')),content_chars=len(content),reasoning_chars=len(reason),chunks=chunks)
 except Exception as e:
  rec.update(error=str(e),duration_s=time.perf_counter()-t)
  if hasattr(e,'read'):rec['error_body']=e.read().decode()[:2000]
 name=f'{phase}-{i}-{rep}-c{c}-{profile}-think{int(thinking)}'
 (out/(name+'.json')).write_text(json.dumps({'metrics':rec,'content':content,'reasoning':reason,'usage':usage},ensure_ascii=False))
 print(json.dumps(rec),flush=True)
 return rec
ids=['short1','short2','medium2','long1']
if len(sys.argv)>2 and sys.argv[2]=='smoke':
 for think in [False,True]:run('short1','smoke',thinking=think,profile='official')
 sys.exit()
run('short1','warmup');flush()
for rep in range(3):
 for i in ids:
  flush();run(i,'cold',rep);run(i,'warm',rep)
for rep in range(3):
 flush()
 for phase in ['cold','warm']:
  t=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rs=list(ex.map(lambda i:run(i,phase,rep,4),ids))
  print(json.dumps({'event':'batch','variant':variant,'phase':phase,'repeat':rep,'concurrency':4,'duration_s':time.perf_counter()-t,'output_tokens':sum(r.get('output_tokens') or 0 for r in rs)}),flush=True)
for profile in ['goobot','official']:
 for think in [False,True]:
  for i in sorted(samples):
   flush();run(i,'quality',thinking=think,profile=profile)

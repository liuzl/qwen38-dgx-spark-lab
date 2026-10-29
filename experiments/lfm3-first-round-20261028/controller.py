import subprocess,pathlib,time,urllib.request,json,os,signal
os.umask(0o077)
p=pathlib.Path('/workspace/lfm3-eval');private=pathlib.Path('/root/lfm3-private')
def healthy():
 try:
  with urllib.request.urlopen('http://127.0.0.1:30000/health',timeout=2) as r:return r.status==200
 except Exception:return False
def stop():
 if (p/'engine.pid').exists():
  pid=int((p/'engine.pid').read_text());subprocess.run(['kill','-TERM',str(pid)],check=False)
  for _ in range(60):
   if not pathlib.Path('/proc/'+str(pid)).exists():break
   time.sleep(1)
  subprocess.run(['pkill','-TERM','-f','sglang::'],check=False)
  time.sleep(5)
# Preserve exploratory run, then repeat with cache accounting and fixed state pool.
while subprocess.run(['pgrep','-f','^python bench.py bf16$'],capture_output=True).returncode==0:time.sleep(2)
stop()
for f in ['bf16','bf16-results.jsonl']:
 if (private/f).exists():(private/f).rename(private/(f+'-pilot'))
if (p/'engine-bf16.log').exists():(p/'engine-bf16.log').rename(p/'engine-bf16-pilot.log')
for v in ['bf16','fp8']:
 print('starting',v,time.time(),flush=True)
 subprocess.run(['bash',str(p/'start.sh'),v],check=True)
 deadline=time.time()+300
 while not healthy():
  if time.time()>deadline:raise RuntimeError('startup timed out '+v)
  time.sleep(3)
 # Readiness endpoint is followed by startup warmup; allow readiness traffic to settle.
 time.sleep(3)
 print('healthy',v,time.time(),flush=True)
 with open(private/(v+'-results.jsonl'),'w') as out:
  r=subprocess.run(['python',str(p/'bench.py'),v],stdout=out,stderr=subprocess.STDOUT)
 print('bench_exit',v,r.returncode,time.time(),flush=True)
 if r.returncode:raise RuntimeError('bench failed '+v)
 stop()
print('complete',time.time(),flush=True)

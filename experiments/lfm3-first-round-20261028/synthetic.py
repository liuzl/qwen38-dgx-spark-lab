import json,pathlib,os
os.umask(0o077)
p=pathlib.Path('/root/lfm3-private/samples')
sys='You play Mira, a friendly fictional bookstore clerk. Address the latest user message. Keep replies under 100 words. Put actions in *italics* and dialogue in double quotes. Do not invent shared history, speak for the user, or decide the user\'s actions. Preserve established facts. Ask for clarification when needed.'
cases={
 'synthetic_state':[{'role':'user','content':'I already paid for the blue notebook. Please put it in a paper bag.'},{'role':'assistant','content':'"Your blue notebook is paid for." *Mira places it in a paper bag.*'},{'role':'user','content':'Thanks. Where is the exit?'}],
 'synthetic_latest':[{'role':'user','content':'Can you recommend a mystery novel?'},{'role':'assistant','content':'"Do you prefer a cozy mystery or something darker?"'},{'role':'user','content':'Actually forget the book. I only need to know when you close.'}],
 'synthetic_agency':[{'role':'user','content':'I am undecided about staying for the reading. What are my options?'}],
 'synthetic_format':[{'role':'user','content':'Tell me you found my lost umbrella. Use one short action and one short spoken sentence.'}],
 'synthetic_chinese':[{'role':'user','content':'请用中文回复。我刚才已经买了笔记本，现在只想问有没有纸袋。不要替我描述动作。'}],
}
for i,msgs in cases.items():(p/(i+'.json')).write_text(json.dumps({'id':i,'messages':[{'role':'system','content':sys}]+msgs,'temperature':.8},ensure_ascii=False))

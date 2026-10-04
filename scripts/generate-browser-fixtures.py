#!/usr/bin/env python3
"""Generate synthetic, distinct UI screenshots for qualify-multi-image.py (requires Pillow)."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--font');args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype(args.font,64) if args.font else ImageFont.load_default(size=64)
for i in range(16):
    im=Image.new('RGB',(1024,768),'#eff3f8');d=ImageDraw.Draw(im)
    d.rectangle((0,0,1024,100),fill='#183553');d.text((35,15),'Synthetic browser page',font=font,fill='white')
    d.text((80,200),f'Page {i+1:02d}',font=font,fill='#183553');d.text((80,350),f'CODE {7400+i}',font=font,fill='black')
    d.rectangle((80,540,800,640),outline='#607d9b',width=4);d.text((100,550),'Continue',font=font,fill='#183553')
    im.save(args.output/f'{i+1:02d}.png')

from service import load_audio, find_possible_watermark_start
from glob import glob
import argparse as ag
import pandas as pd
import os
import math

def format_list_time(list):
    return [format_time(item) for item in list]

def format_time(ms: float) -> str:
    remainder = int(ms)
    sec_rem = ms - remainder
    hours = remainder // 3600
    remainder = remainder % 3600
    minutes = remainder // 60
    remainder %= 60
    seconds = remainder + sec_rem
    return f"{hours:02d}:{minutes:02d}:{seconds:.1f}"

if __name__ == "__main__":

    parser = ag.ArgumentParser()
    parser.add_argument('--id', type=str)
    parser.add_argument('--path', type=str, required=True,
                        help='base folder path to find')
    parser.add_argument('--topk', type=int, default=2,
                        help='number of possible occurrences')
    parser.add_argument('--sr', type=int, default=8000,
                        help='sample rate of voices to work on')
    args = parser.parse_args()
    if not args.id:
        args.id = args.path

    # Configure paths with your project structure 
    welcome_watermark, _ = load_audio('watermark_1.mp3', args.sr)
    goodbye_watermark, _ = load_audio('watermark_2.mp3', args.sr)

    data = []
    if os.path.isfile('watermark_timestamps.csv'):
        data = pd.read_csv('watermark_timestamps.csv').values.tolist()
        
    case, _ = load_audio(args.path, args.sr)
    welcome_case = case[:10 * 60 * args.sr]
    goodbye_case = case[len(case) - 10 * 60 * args.sr:]
    
    welcome_watermark_start = find_possible_watermark_start(welcome_watermark, welcome_case, top_k=args.topk, sr=args.sr)
    goodbye_watermark_start = [len(case) / args.sr - 10 * 60 + pt for pt in find_possible_watermark_start(goodbye_watermark, goodbye_case, top_k=args.topk, sr=args.sr)]
    
    data.append([args.id, args.path] + format_list_time(welcome_watermark_start + goodbye_watermark_start))

    pd.DataFrame(data, columns=['id', 'filepath'] +
                 [f'p{i + 1}' for i in range(args.topk * 2)]).to_csv('watermark_timestamps.csv', index=False)

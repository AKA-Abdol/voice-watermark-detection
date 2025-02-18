from service import load_audio, find_possible_watermark_start
from glob import glob
import argparse as ag
import pandas as pd

if __name__ == "__main__":

    parser = ag.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help='base folder path to find')
    parser.add_argument('--topk', type=int, default=2,
                        help='number of possible occurrences')
    parser.add_argument('--sr', type=int, default=8000,
                        help='sample rate of voices to work on')
    args = parser.parse_args()

    welcome_watermark, _ = load_audio('watermark_1.mp3', args.sr)
    goodbye_watermark, _ = load_audio('watermark_2.mp3', args.sr)

    data = []
    for case_path in glob('**/*.mp4', root_dir=args.path, recursive=True):
        case, _ = load_audio(case_path, args.sr)
        welcome_case = case[:10 * 60 * 1000]
        goodbye_case = case[len(case) - 10 * 60 * 1000:]
        data.append([case_path] + find_possible_watermark_start(welcome_watermark, welcome_case, top_k=args.topk, sr=args.sr) +
                    find_possible_watermark_start(goodbye_watermark, goodbye_case, top_k=args.topk, sr=args.sr))

    pd.DataFrame(data, columns=['filepath'] +
                 [f'p{i + 1}' for i in range(args.topk * 2)]).to_csv('watermark_timestamps.csv', index=False)

"""torchrun entry point: isolate each worker to its assigned physical GPU."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if visible:
    devices = visible.split(',')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if local_rank >= len(devices):
        raise RuntimeError('Each distributed worker requires a distinct allowed GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('LOCAL_RANK', '0')

from train import parser, run

if __name__ == '__main__':
    options = parser().parse_args()
    if options.backend != 'warp':
        raise ValueError('Distributed training requires --backend warp')
    options.device = 'cuda:0'
    run(options)

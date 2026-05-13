import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import argparse
from pathlib import Path

import torch
import yaml

def load_config(baseline_path, dataset_override=None):
    with open(baseline_path) as f:
        config = yaml.safe_load(f)
    if dataset_override:
        config['dataset'] = dataset_override.upper()
    dataset = config.get('dataset', '').upper()
    if not dataset:
        raise ValueError("Config must contain a 'dataset' key (DEAP or SEEDVIG).")
    dataset_yaml = Path(baseline_path).parent / f'{dataset}.yaml'
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {dataset_yaml}\n"
            f"Expected configs/{dataset}.yaml"
        )
    with open(dataset_yaml) as f:
        config.update(yaml.safe_load(f))
    config['dataset'] = dataset
    return config


def apply_work_dir(config, work_dir):
    wd = Path(work_dir)
    config['saved_model_dir']  = str(wd / config.get('saved_model_dir',  'bin'))
    config['csv_log_save_dir'] = str(wd / config.get('csv_log_save_dir', 'csv_logs'))
    config['results_dir']      = str(wd / config.get('results_dir',      'results'))


def save_config_snapshot(config, work_dir):
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    with open(wd / 'config.yaml', 'w') as f:
        yaml.dump(config, f)


def set_device_env(device_str):
    if str(device_str) in ('-1', 'cpu', ''):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return -1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_str)
    return int(str(device_str).split(',')[0])


def get_parser():
    p = argparse.ArgumentParser(
        description='EfficientCapsNet EEG - unified train / test / features entry point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--phase',        default='train',
                   choices=['train', 'test', 'features'],
                   help='train: K-fold training; test: inference with metrics; '
                        'features: extract capsule vectors to .npy')
    p.add_argument('--config',       default='configs/baseline.yaml',
                   help='Path to baseline YAML config')
    p.add_argument('--dataset',      default=None,
                   choices=['deap', 'DEAP', 'seedvig', 'SEEDVIG'],
                   help='Dataset override (default: value in config)')
    p.add_argument('--work-dir',     default=None,
                   help='Root output directory; redirects saved_model_dir / '
                        'csv_log_save_dir / results_dir under this path')
    p.add_argument('--device',       default='0',
                   help='GPU id(s): 0 | 0,1 | -1 for CPU')
    p.add_argument('--load-weights', default=None,
                   help='Path to .pt weight file (required for --phase test / features)')
    p.add_argument('--mode',         default='sub_independent',
                   choices=['sub_independent', 'sub_dependent'],
                   help='Training / inference mode for DEAP')

    # DEAP-specific
    p.add_argument('--dimension',    default=None,
                   choices=['VALENCE', 'AROUSAL', 'DOMINANCE', 'LIKING', 'VA', 'VAD'],
                   help='Emotion dimension for DEAP test / features phase')
    p.add_argument('--subject',      type=int, default=None,
                   help='1-indexed subject number (DEAP sub_dependent only)')

    # SEED-VIG-specific
    p.add_argument('--session',      default=None,
                   help='Single session name for SEED-VIG '
                        '(e.g. 1_20151124_noon_2); omit to pool all sessions')

    p.add_argument('--batch-size',   type=int, default=64,
                   help='Batch size for inference / feature extraction')
    return p


if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()

    dev_id = set_device_env(args.device)
    config = load_config(args.config, dataset_override=args.dataset)
    config['device'] = dev_id

    work_dir = args.work_dir or config.get('work_dir')
    if work_dir:
        apply_work_dir(config, work_dir)
        save_config_snapshot(config, work_dir)
        args.work_dir = work_dir   # propagate to eeg_features / eeg_eval

    device = torch.device('cuda' if torch.cuda.is_available() and dev_id != -1 else 'cpu')
    print(f'Device  : {device}')
    print(f'Dataset : {config["dataset"]}')
    print(f'Phase   : {args.phase}')

    from eeg_scripts import eeg_train, eeg_eval, eeg_features

    if args.phase == 'train':
        eeg_train(config, mode=args.mode)

    elif args.phase == 'test':
        if args.load_weights is None:
            parser.error('--load-weights PATH is required for --phase test')
        eeg_eval(args, config, device)

    elif args.phase == 'features':
        if args.load_weights is None:
            parser.error('--load-weights PATH is required for --phase features')
        eeg_features(args, config, device)

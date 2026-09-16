"""Run independent category experiments through the two training stages."""
import argparse
from copy import deepcopy
from pathlib import Path
import shlex
import subprocess
import sys

import yaml
from amazon2014 import CATEGORIES, canonical_category

STAGES = ('prepare', 'semantic', 'export', 'recommendation', 'evaluate')


def plan(config, categories, raw_dir, data_root, output_root, num_gpus=1, download=False, stages=STAGES):
    jobs = []
    for category in categories:
        cfg = deepcopy(config)
        run_dir = Path(output_root) / category
        cfg['categories'] = [category]
        cfg['dataset_version'] = '2014'
        cfg['data_dir'] = str(Path(data_root) / category)
        cfg['trainer']['output_dir'] = str(run_dir / 'semantic')
        cfg['recommendation']['trainer']['output_dir'] = str(run_dir / 'recommendation')
        cfg_path = run_dir / 'experiment.yaml'
        semantic = run_dir / 'semantic' / 'final'
        index = run_dir / 'semantic_index.json'
        rec = run_dir / 'recommendation' / 'final'
        commands = {
            'prepare': [sys.executable, 'amazon2014.py', '--categories', category,
                '--raw-dir', str(raw_dir), '--output-dir', str(data_root), *(['--download'] if download else [])],
            'semantic': [sys.executable, 'train.py', '--config-path', str(cfg_path)],
            'export': [sys.executable, 'export_semantic_ids.py', '--config-path', str(cfg_path),
                '--checkpoint', str(semantic), '--output', str(index)],
            'recommendation': [sys.executable, 'train_recommendation.py', '--config-path', str(cfg_path),
                '--checkpoint', str(semantic), '--index', str(index)],
            'evaluate': [sys.executable, 'evaluate_recommendation.py', '--checkpoint', str(rec),
                '--config-path', str(cfg_path)],
        }
        if num_gpus > 1:
            for stage in ('semantic', 'recommendation'):
                commands[stage] = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nnodes=1', f'--nproc_per_node={num_gpus}', *commands[stage][1:]]
        jobs.append((cfg, cfg_path, [(s, commands[s]) for s in stages]))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', default='configs.yaml')
    parser.add_argument('--categories', nargs='+')
    parser.add_argument('--raw-dir', default='data/amazon2014/raw')
    parser.add_argument('--data-root', default='data/amazon2014/processed')
    parser.add_argument('--output-root')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error('--num-gpus must be positive')
    config = yaml.safe_load(Path(args.config_path).read_text())
    categories = list(dict.fromkeys(map(canonical_category, args.categories or config.get('categories', CATEGORIES))))
    jobs = plan(config, categories, args.raw_dir, args.data_root,
                args.output_root or config['output_root'], args.num_gpus, args.download, args.stages)
    if not args.dry_run and any(s in args.stages for s in ('semantic', 'recommendation')):
        import torch
        if not torch.cuda.is_available() and not config['trainer'].get('use_cpu', False):
            parser.error('CUDA is unavailable. Use --dry-run here and execute training on the GPU host.')
    for cfg, path, commands in jobs:
        print(f"\nCategory: {cfg['categories'][0]}; configuration: {path}", flush=True)
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and yaml.safe_load(path.read_text()) != cfg:
                raise ValueError(f'Existing experiment config differs: {path}; choose a new output root')
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        for stage, command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                if stage in ('semantic', 'recommendation'):
                    directory = cfg['trainer']['output_dir'] if stage == 'semantic' else cfg['recommendation']['trainer']['output_dir']
                    if Path(directory, 'final').exists():
                        raise FileExistsError(f'{directory}/final already exists; select remaining --stages or a new output root')
                subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

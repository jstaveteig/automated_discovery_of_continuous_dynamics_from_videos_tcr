import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('single_pendulum', 'double_pendulum')
SEEDS = (1, 2, 3)
TANGENT = {
    'model_name': 'smoothTC',
    'tangent_step_weight': 1.0,
    'tangent_loss_weight': 2.0,
    'tangent_norm_weight': 1.0,
    'tangent_angle_weight': 1.0,
    'tangent_k': 8,
    'tangent_warmup_epochs': 25,
    'tangent_ramp_epochs': 20,
    'tangent_eps': 1.0e-6,
}


def smooth_name(config):
    return '_'.join([
        config['model_name'],
        str(config['seed']),
        config['reconstruct_loss_type'],
        str(config['reconstruct_loss_weight']),
        config['smooth_loss_type'],
        str(config['smooth_loss_weight']),
        config['regularize_loss_type'],
        str(config['regularize_loss_weight']),
        str(config['annealing']),
    ])


def run(command, done_path, env, dry_run, force):
    if done_path.exists() and not force:
        print(f'skip {done_path.stem}')
        return
    print(' '.join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.touch()


def main():
    parser = argparse.ArgumentParser(description='run the smooth vs smoothTC ablation')
    parser.add_argument('--run-name', default='tangent_ablation')
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip-stage1', action='store_true')
    args = parser.parse_args()

    run_root = ROOT / 'runs' / args.run_name
    config_root = run_root / 'runtime_configs'
    done_root = run_root / '.done'
    output_dir = (run_root / 'outputs').relative_to(ROOT).as_posix()
    config_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault('WANDB_MODE', 'offline')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    python = sys.executable
    for dataset in DATASETS:
        for seed in SEEDS:
            if not args.skip_stage1:
                stage1 = yaml.safe_load((ROOT / 'configs' / dataset / f'trial{seed}' / 'encoder-decoder-64.yaml').read_text())
                stage1.update({'seed': seed, 'dataset': dataset, 'output_dir': output_dir})
                stage1_path = config_root / f'{dataset}_trial{seed}_stage1.yaml'
                stage1_path.parent.mkdir(parents=True, exist_ok=True)
                stage1_path.write_text(yaml.safe_dump(stage1, sort_keys=False))
                run([python, 'main.py', '-config', str(stage1_path), '-mode', 'train'], done_root / f'{dataset}_seed{seed}_stage1_train.ok', env, args.dry_run, args.force)

            for variant in ('smooth', 'smoothTC'):
                stage2 = yaml.safe_load((ROOT / 'configs' / dataset / f'trial{seed}' / 'smooth.yaml').read_text())
                stage2.update({'seed': seed, 'dataset': dataset, 'output_dir': output_dir, 'model_name': variant})
                if variant == 'smoothTC':
                    stage2.update(TANGENT)
                stage2_path = config_root / f'{dataset}_trial{seed}_{variant}.yaml'
                stage2_path.write_text(yaml.safe_dump(stage2, sort_keys=False))
                run([python, 'main.py', '-config', str(stage2_path), '-mode', 'train'], done_root / f'{dataset}_seed{seed}_{variant}_train.ok', env, args.dry_run, args.force)
                run([python, 'main.py', '-config', str(stage2_path), '-mode', 'test_all'], done_root / f'{dataset}_seed{seed}_{variant}_test_all.ok', env, args.dry_run, args.force)

                stage3 = yaml.safe_load((ROOT / 'configs' / dataset / f'trial{seed}' / 'regress-smooth-filtered.yaml').read_text())
                stage3.update({'seed': seed, 'dataset': dataset, 'output_dir': output_dir, 'nsv_model_name': smooth_name(stage2), 'filter_data': True})
                stage3_path = config_root / f'{dataset}_trial{seed}_{variant}_regress.yaml'
                stage3_path.write_text(yaml.safe_dump(stage3, sort_keys=False))
                run([python, 'regress.py', '-config', str(stage3_path), '-mode', 'train'], done_root / f'{dataset}_seed{seed}_{variant}_regress_train.ok', env, args.dry_run, args.force)
                run([python, 'regress.py', '-config', str(stage3_path), '-mode', 'test'], done_root / f'{dataset}_seed{seed}_{variant}_regress_test.ok', env, args.dry_run, args.force)
if __name__ == '__main__':
    main()

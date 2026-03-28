from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from eval import evaluate_candidate_vs_reference, evaluate_method_comparison
from train_rl import run_dpo, run_grpo, run_ppo
from utils.config import load_config
from utils.io import ensure_dir, make_run_dir, save_json


def _load_metrics_frame(run_dir: str | Path) -> pd.DataFrame:
    path = Path(run_dir) / 'metrics.jsonl'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _run_variant(config: dict, ablation: str, label: str) -> dict:
    if ablation in {'kl_sweep', 'clip_sweep'}:
        method = config['ablation']['method']
        if method == 'ppo':
            return run_ppo(config)
        if method == 'grpo':
            return run_grpo(config)
        raise ValueError('kl_sweep/clip_sweep require method=ppo or method=grpo')
    if ablation == 'k_sweep':
        return run_grpo(config)
    if ablation == 'dpo_beta_sweep':
        return run_dpo(config)
    raise ValueError(f'Unsupported ablation: {ablation}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--ablation', choices=['kl_sweep', 'clip_sweep', 'k_sweep', 'dpo_beta_sweep'], required=True)
    parser.add_argument('--method', choices=['ppo', 'grpo'], default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_config.setdefault('ablation', {})
    if args.method is not None:
        base_config['ablation']['method'] = args.method
    reference_checkpoint = base_config['models'].get('sft_checkpoint') or base_config['models'].get('policy_checkpoint')
    if not reference_checkpoint:
        raise ValueError('SFT checkpoint is required before running ablations.')

    run_dir = make_run_dir(base_config['output_dir'], f'ablation_{args.ablation}')
    results = []

    if args.ablation == 'kl_sweep':
        values = [0.0, 0.05, 0.1, 0.5]
        method = base_config['ablation']['method']
        for beta in values:
            config = deepcopy(base_config)
            config[method]['beta'] = beta
            result = _run_variant(config, args.ablation, f'beta_{beta}')
            metrics = evaluate_candidate_vs_reference(config, result['policy_checkpoint'], reference_checkpoint)
            row = {
                'method': method,
                'beta': beta,
                'run_dir': result['run_dir'],
                'mean_rm_score': metrics['mean_candidate_rm_score'],
                'mean_kl': metrics['mean_sampled_token_kl'],
                'rm_win_rate_vs_sft': metrics['reward_model_win_rate_vs_sft'],
            }
            results.append(row)

    elif args.ablation == 'clip_sweep':
        values = [0.05, 0.2, 0.5, float('inf')]
        method = base_config['ablation']['method']
        for eps in values:
            config = deepcopy(base_config)
            config[method]['epsilon'] = 1.0e9 if eps == float('inf') else eps
            result = _run_variant(config, args.ablation, f'eps_{eps}')
            metrics = evaluate_candidate_vs_reference(config, result['policy_checkpoint'], reference_checkpoint)
            frame = _load_metrics_frame(result['run_dir'])
            row = {
                'method': method,
                'epsilon': 'inf' if eps == float('inf') else eps,
                'run_dir': result['run_dir'],
                'mean_rm_score': metrics['mean_candidate_rm_score'],
                'mean_kl': metrics['mean_sampled_token_kl'],
                'reward_variance': float(frame['mean_reward'].var()) if 'mean_reward' in frame else None,
                'grad_norm_variance': float(frame.filter(regex='grad_norm').mean(axis=1).var()) if not frame.empty else None,
            }
            results.append(row)

    elif args.ablation == 'k_sweep':
        base_calls = base_config['grpo']['prompts_per_step'] * base_config['grpo']['k_rollouts']
        for k in [1, 2, 4, 8]:
            config = deepcopy(base_config)
            config['grpo']['k_rollouts'] = k
            config['grpo']['prompts_per_step'] = max(1, base_calls // k)
            result = _run_variant(config, args.ablation, f'k_{k}')
            metrics = evaluate_candidate_vs_reference(config, result['policy_checkpoint'], reference_checkpoint)
            frame = _load_metrics_frame(result['run_dir'])
            row = {
                'k_rollouts': k,
                'prompts_per_step': config['grpo']['prompts_per_step'],
                'run_dir': result['run_dir'],
                'mean_rm_score': metrics['mean_candidate_rm_score'],
                'mean_kl': metrics['mean_sampled_token_kl'],
                'degenerate_fraction': float(frame['degenerate'].mean()) if 'degenerate' in frame else None,
            }
            results.append(row)

    elif args.ablation == 'dpo_beta_sweep':
        for beta in [0.01, 0.1, 0.5, 1.0]:
            config = deepcopy(base_config)
            config['dpo']['beta'] = beta
            result = _run_variant(config, args.ablation, f'beta_{beta}')
            metrics = evaluate_candidate_vs_reference(config, result['policy_checkpoint'], reference_checkpoint)
            row = {
                'beta': beta,
                'run_dir': result['run_dir'],
                'mean_rm_score': metrics['mean_candidate_rm_score'],
                'mean_kl': metrics['mean_sampled_token_kl'],
                'heldout_preference_accuracy': metrics['heldout_preference_accuracy'],
                'rm_win_rate_vs_sft': metrics['reward_model_win_rate_vs_sft'],
            }
            results.append(row)

    save_json(run_dir / 'ablation_results.json', {'ablation': args.ablation, 'results': results})
    pd.DataFrame(results).to_csv(run_dir / 'ablation_results.csv', index=False)


if __name__ == '__main__':
    main()

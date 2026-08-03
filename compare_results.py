"""
Compare results from parameter sweep.
Run after run_sweep.sh has completed.

Usage:
    python compare_results.py
    python compare_results.py --results_dir my_results_folder
"""

import json
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.results_dir}/*.json"))

if not files:
    print(f"No JSON files found in {args.results_dir}/")
    exit(1)

rows = []
for path in files:
    with open(path) as f:
        r = json.load(f)
    rows.append({
        'file':          os.path.basename(path),
        'final_mean_20': r['final_mean_20'],
        'best_episode':  r['best_episode'],
        'mean_all':      r['mean_all'],
        'config':        r['config'],
    })

# sort by final mean descending — best configuration first
rows.sort(key=lambda x: x['final_mean_20'], reverse=True)

print(f"\n{'Rank':<6} {'File':<35} {'Final mean 20':>14} {'Best episode':>13} {'Mean all':>10}")
print("-" * 80)
for i, row in enumerate(rows):
    print(f"{i+1:<6} {row['file']:<35} {row['final_mean_20']:>14.1f} {row['best_episode']:>13} {row['mean_all']:>10.1f}")

print(f"\n{'='*80}")
print("Top configuration:")
print(f"  File: {rows[0]['file']}")
print(f"  Final mean (last 20 episodes): {rows[0]['final_mean_20']:.1f}")
print(f"  Best episode: {rows[0]['best_episode']} steps")
print(f"  Parameters:")
for k, v in rows[0]['config'].items():
    if k != 'output_file':
        print(f"    {k}: {v}")

print(f"\n{'='*80}")
print("Bottom configuration:")
print(f"  File: {rows[-1]['file']}")
print(f"  Final mean (last 20 episodes): {rows[-1]['final_mean_20']:.1f}")
print(f"  Parameters:")
for k, v in rows[-1]['config'].items():
    if k != 'output_file':
        print(f"    {k}: {v}")

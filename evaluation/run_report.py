"""Compare method directories emitted by watch-wisely; gold stays outside inference."""
import argparse
import json
from pathlib import Path
from statistics import mean

from .compare import _load
from .compare_records import compare_records
from .summarize_calls import summarize_call_records
from watch_wisely.experiment import METHODS, sha256_file


def build_report(run_path, manifest_path, baseline='Uniform-24', adaptive='Adaptive-24', samples=10000):
    run_path = Path(run_path)
    metadata = json.loads((run_path / 'run.json').read_text(encoding='utf-8'))
    if metadata.get('synthetic') is not False:
        raise ValueError('synthetic runs cannot be scored')
    gold, gold_sha = _load(Path(manifest_path))
    records = {}
    for method in (baseline, adaptive):
        if method not in METHODS or method not in metadata['methods']:
            raise ValueError('method missing from run')
        records[method] = [json.loads(p.read_text(encoding='utf-8'))
                           for p in sorted((run_path / method).glob('*.json'))]
    expected = {q['question_id']: q['video_id'] for q in metadata['questions']}
    if len(expected) != len(metadata['questions']):
        raise ValueError('duplicate IDs in run metadata')
    if {q['question_id']: q['video_id'] for q in gold} != expected:
        raise ValueError('gold manifest differs from run question/video IDs')
    report = compare_records(gold, records[baseline], records[adaptive],
        baseline_run_id=metadata['run_id'], adaptive_run_id=metadata['run_id'],
        baseline_method=baseline, adaptive_method=adaptive, bootstrap_samples=samples)
    costs = {}
    for method, rows in records.items():
        results = [r['result'] for r in rows]
        calls = [call for r in results for call in r['calls']]
        costs[method] = dict(
            mean_observed_frames=mean(sum(len(o['frame_ids']) for o in r['observations']) for r in results),
            mean_question_latency_ms=mean(r['wall_latency_ms'] for r in results),
            mean_calls=mean(len(r['calls']) for r in results),
            calls=summarize_call_records(calls) if calls else None)
    report.update(costs=costs, run_metadata_sha256=sha256_file(run_path / 'run.json'),
                  gold_manifest_sha256=gold_sha,
                  note='Failures remain incorrect in the denominator. Provenance is recorded, not independently certified.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--manifest', required=True, help='scoring JSON array with question_id, video_id, answer')
    parser.add_argument('--baseline', choices=METHODS, default='Uniform-24')
    parser.add_argument('--adaptive', choices=METHODS, default='Adaptive-24')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args(argv)
    report = build_report(args.run, args.manifest, args.baseline, args.adaptive, args.bootstrap_samples)
    print(json.dumps(report, allow_nan=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

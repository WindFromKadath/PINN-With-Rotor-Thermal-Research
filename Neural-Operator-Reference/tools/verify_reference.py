"""Replay every local reference and saved run, and verify migration hashes."""
import argparse
import json
from pathlib import Path
import numpy as np
from heatops.cli.evaluate_run import main as evaluate_run
from heatops.cli.legacy_evaluate import main as evaluate_legacy
from heatops.training.run_context import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--source', help='Optional read-only original directory for SHA-256 verification')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'usability and provenance; no accuracy or convergence claim', 'legacy': [], 'runs': [], 'errors': []}
    local = root / 'local_reference_versions'
    copied = json.loads((local / 'copied_files.json').read_text(encoding='utf-8'))
    bad_copies = [r['destination'] for r in copied if not (root / r['destination']).exists() or file_hash(root / r['destination']) != r['sha256']]
    report['copied_files'] = {'count': len(copied), 'mismatches': bad_copies}
    if args.source:
        source = Path(args.source).resolve()
        inventory = json.loads((local / 'source_inventory.json').read_text(encoding='utf-8'))
        mismatches = [r['source'] for r in inventory if not (source / r['source']).exists() or file_hash(source / r['source']) != r['sha256']]
        report['original_files'] = {'count': len(inventory), 'mismatches': mismatches}
    for package in json.loads((local / 'index.json').read_text(encoding='utf-8'))['packages']:
        spec = json.loads((local / package / 'model_spec.json').read_text(encoding='utf-8'))
        for model in spec['models']:
            dest = out / 'legacy' / f'{package}_{model}'
            try:
                evaluate_legacy(['--package', str(local / package), '--model', model, '--out-dir', str(dest)])
                report['legacy'].append({'package': package, **json.loads((dest / 'metrics.json').read_text(encoding='utf-8'))})
            except Exception as exc:
                report['errors'].append({'legacy': f'{package}/{model}', 'error': str(exc)})
    for cfg in sorted((root / 'artifacts').glob('E*/**/config.json')):
        rd = cfg.parent
        if not (rd / 'metrics.json').exists():
            continue
        label = str(rd.relative_to(root / 'artifacts')).replace('\\', '/').replace('/', '_')
        dest = out / 'runs' / label
        try:
            evaluate_run(['--run-dir', str(rd), '--out-dir', str(dest)])
            metrics = json.loads((dest / 'metrics.json').read_text(encoding='utf-8'))
            report['runs'].append({'run': label, **metrics})
        except Exception as exc:
            report['errors'].append({'run': label, 'error': str(exc)})
    smoke = root / 'artifacts' / 'E108' / 'predictions_eval.npz'
    if smoke.exists():
        with np.load(smoke) as bundle:
            times = bundle['t_star']
            report['initial_frame'] = {'length': len(times), 'zero_count': int(np.count_nonzero(times == 0)), 'strictly_increasing': bool(np.all(np.diff(times) > 0))}
    report['passed'] = not (bad_copies or report['errors'] or report.get('original_files', {}).get('mismatches'))
    write_json(out / 'verification.json', report)
    print(json.dumps({'passed': report['passed'], 'legacy_models': len(report['legacy']), 'runs': len(report['runs']), 'errors': report['errors']}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

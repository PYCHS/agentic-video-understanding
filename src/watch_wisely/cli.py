"""Command-line entry point. Heavy inference imports occur only after preflight."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform

from .experiment import METHODS, execute, load_questions, sha256_file
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .qwen import QwenBackend, QwenSettings


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run frozen-Qwen video QA with saved evidence and failures.')
    parser.add_argument('--manifest', required=True, help='JSON array; relative video paths resolve beside it')
    parser.add_argument('--settings', required=True, help='QwenSettings JSON, including immutable revisions')
    parser.add_argument('--output', required=True, help='new output directory; existing paths are refused')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['Uniform-24', 'Adaptive-24'])
    parser.add_argument('--allow-download', action='store_true', help='permit Hugging Face weight downloads')
    parser.add_argument('--validate-only', action='store_true', help='preflight without loading models or writing files')
    args = parser.parse_args(argv)
    questions = load_questions(args.manifest)
    settings = QwenSettings(**json.loads(Path(args.settings).read_text(encoding='utf-8')))
    if not args.run_id.strip() or len(set(args.methods)) != len(args.methods):
        parser.error('run ID must be nonempty and methods must be unique')
    if Path(args.output).exists():
        parser.error('output directory already exists; use a new path')
    if args.validate_only:
        print(json.dumps(dict(questions=len(questions), methods=args.methods,
                              model=QwenBackend.MODEL_ID, settings=asdict(settings))))
        return 0
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'av', 'Pillow'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Hash implementation files as well as manifests to make each run inspectable.
    source = Path(__file__).parent
    provenance = dict(model=QwenBackend.MODEL_ID, settings=asdict(settings),
        prompt_version=PROMPT_VERSION,
        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        manifest_sha256=sha256_file(args.manifest), packages=packages,
        python=platform.python_version(), platform=platform.platform(),
        source_sha256={p.name: sha256_file(p) for p in sorted(source.glob('*.py'))})
    backend = QwenBackend.load(settings, local_files_only=not args.allow_download)
    counts = execute(questions, backend, args.output, run_id=args.run_id,
                     methods=args.methods, provenance=provenance)
    print(json.dumps(counts))
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

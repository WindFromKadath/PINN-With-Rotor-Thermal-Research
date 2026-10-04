"""Check this project's Git submission scope without creating or changing Git."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import unquote

ROOT_FILES = {'LICENSE', '.gitignore', '.python-version', 'README.md', 'README.zh-CN.md',
              'AGENTS.md', 'MAINTAINER.md', 'CONTRIBUTING.md', 'CONTRIBUTING.zh-CN.md',
              'pyproject.toml', 'uv.lock', 'hypotheses.csv', 'experiments.csv',
              'requirements.txt', 'requirements-lock.txt'}
DIRECTORIES = {'src', 'configs', 'docs', 'tests', 'tools'}
SKIP = {'.git', '.venv', '.venv-cuda', '.cache', '.pytest_cache', '.ruff_cache',
        '__pycache__', 'node_modules', 'local_reference_versions', 'artifacts',
        'build', 'dist', '.idea', '.vscode', '.ipynb_checkpoints', 'private-audit', 'raw-sessions'}
PRIVATE_NAMES = {'credentials.json', 'auth.json', '.privacy.local.json'}
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.mph', '.stl', '.pdf', '.docx', '.pptx', '.zip', '.7z'}
FILE_URI = 'file:' + '//'
DRIVE = re.compile(r'(?<!\w)[A-Z]:[/\\]', re.I)
LINK = re.compile(r'!?\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"\n]*")?\)')


def candidates(root):
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        for parent, dirs, names in os.walk(root / name):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.endswith('.egg-info'))
            for filename in sorted(names):
                p = Path(parent) / filename
                if filename in PRIVATE_NAMES or filename.startswith('privacy-report'):
                    continue
                if filename.startswith('.env') and filename != '.env.example':
                    continue
                if p.suffix in {'.pyc', '.pyo', '.pem', '.key'}:
                    continue
                files.append(p)
    for rel in ('data/README.md', 'data/raw/.gitkeep', 'data/processed/.gitkeep'):
        if (root / rel).is_file():
            files.append(root / rel)
    return sorted(set(files), key=lambda p:p.relative_to(root).as_posix())


def check(root, private_tokens=()):
    root = root.resolve()
    files = candidates(root)
    selected = {p.resolve() for p in files}
    issues = []
    links = 0
    parsed = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            issues.append({'file':rel, 'kind':'symlink_or_external_file'})
            continue
        if p.suffix in FORBIDDEN or p.stat().st_size > 10 * 1024 * 1024:
            issues.append({'file':rel, 'kind':'binary_or_oversize_file'})
            continue
        try:
            text = p.read_text(encoding='utf-8-sig')
        except UnicodeError:
            issues.append({'file':rel, 'kind':'non_utf8_file'})
            continue
        if DRIVE.search(text) or FILE_URI in text:
            issues.append({'file':rel, 'kind':'machine_absolute_path'})
        if any(token and token in text for token in private_tokens):
            issues.append({'file':rel, 'kind':'private_token'})
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}', text):
            issues.append({'file':rel, 'kind':'credential_pattern'})
        try:
            if p.suffix == '.py':
                ast.parse(text, filename=rel)
                parsed += 1
            elif p.suffix == '.json':
                json.loads(text)
            elif p.suffix == '.toml' or p.name == 'uv.lock':
                tomllib.loads(text)
        except (ValueError, SyntaxError):
            issues.append({'file':rel, 'kind':'parse_error'})
        if p.suffix != '.md':
            continue
        for m in LINK.finditer(text):
            target = unquote(m[1].strip('<>'))
            if not target or target.startswith('#'):
                continue
            if target.startswith(('http://', 'https://', FILE_URI)):
                issues.append({'file':rel, 'kind':'external_document_reference'})
                continue
            local = (p.parent / target.split('#', 1)[0]).resolve()
            links += 1
            if local not in selected and not (local.is_dir() and any(local in f.parents for f in selected)):
                issues.append({'file':rel, 'kind':'link_outside_project_submission', 'target':target})
    derived = {'docs/publication/git_readiness.json', 'docs/publication/manifest.json'}
    manifest = [{'path':p.relative_to(root).as_posix(), 'bytes':p.stat().st_size,
                 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
                if p.relative_to(root).as_posix() not in derived]
    return {'format_version':1, 'status':'passed' if not issues else 'failed',
            'scope':'this project only; code/configuration/locks/tests/necessary text and metadata',
            'file_count':len(files), 'bytes':sum(p.stat().st_size for p in files),
            'max_file_bytes':max((p.stat().st_size for p in files), default=0),
            'markdown_links_checked':links, 'python_files_parsed':parsed,
            'issues':issues, 'manifest':manifest,
            'excluded':'data contents, checkpoints, historical originals, run artifacts, environments and caches',
            'git_actions':[], 'limits':'Candidate check only; actual staged changes and later additions need their own review.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--private-token', action='append', default=[], help='Optional local token; never printed')
    args = parser.parse_args()
    result = check(args.root, args.private_token)
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}, ensure_ascii=False, indent=2))
    return 0 if not result['issues'] else 1


if __name__ == '__main__':
    sys.exit(main())

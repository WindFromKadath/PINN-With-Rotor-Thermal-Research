"""Read-only checks for the four-project collection; no Git changes."""
import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import unquote

NAMES = ('Neural-Operator-Reference', 'Pinn-Inverse-Rotor-Thickness',
         'Pinn-Ring-Rib-2.5D', 'Pinn-3D-Thermo-Mechanical-Rotor')
ROOT_FILES = ('.gitattributes', '.gitignore', 'README.md', 'README.zh-CN.md', 'LICENSE',
              'AGENTS.md', 'CONTRIBUTING.md', 'CONTRIBUTING.zh-CN.md')

def check(root, private_tokens=()):
    root = Path(root).resolve()
    issues, projects, selected = [], {}, set()
    for i, name in enumerate(NAMES):
        project = root/name
        spec = importlib.util.spec_from_file_location(f'project_check_{i}', project/'tools/check_git_ready.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.check(project, private_tokens)
        projects[name] = {k:v for k,v in result.items() if k!='manifest'}
        issues.extend({'project':name, **row} for row in result['issues'])
        selected.update(module.candidates(project))
        if (project/'LICENSE').read_bytes() != (root/'LICENSE').read_bytes():
            issues.append({'project':name,'kind':'license_mismatch'})
        if (project/'.git').exists():
            issues.append({'project':name,'kind':'nested_git'})
    root_files = [root/name for name in ROOT_FILES if (root/name).is_file()]
    for directory in ('docs','tools'):
        root_files.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    selected.update(root_files)
    selected = {p.resolve() for p in selected}
    for p in root_files:
        rel = p.relative_to(root).as_posix()
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            issues.append({'file':rel,'kind':'external_file'})
            continue
        content = p.read_text(encoding='utf-8-sig')
        if p.stat().st_size > 10*1024*1024:
            issues.append({'file':rel,'kind':'oversize_file'})
        if any(token and token in content for token in private_tokens):
            issues.append({'file':rel,'kind':'private_token'})
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}',content):
            issues.append({'file':rel,'kind':'credential_pattern'})
        if re.search(r'(?<![\w])[A-Z]:[\\/]',content,re.I):
            issues.append({'file':rel,'kind':'machine_path'})
        if p.suffix=='.py': ast.parse(content)
        if p.suffix=='.json': json.loads(content)
        if p.suffix=='.md':
            for raw in re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)\)',content):
                if raw.startswith('#'): continue
                target=(p.parent/unquote(raw.split('#',1)[0])).resolve()
                if target not in selected and not (target.is_dir() and any(target in f.parents for f in selected)):
                    issues.append({'file':rel,'kind':'nonlocal_or_unselected_link','target':raw})
    derived = {'docs/publication/readiness.json', 'docs/publication/manifest.json'}
    manifest = [{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,
                 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(selected) if p.relative_to(root).as_posix() not in derived]
    return {'run_id':'MOVE20261004_01','status':'passed' if not issues else 'failed',
            'license':'MIT','projects':projects,'issues':issues,'manifest':manifest,
            'file_count':len(selected),'bytes':sum(p.stat().st_size for p in selected),
            'git_actions':[], 'scope':'Candidate text/code scope; actual staging and history must be checked before publication.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-token',action='append',default=[],help='Local token to check; never printed')
    args=parser.parse_args()
    result=check(Path(__file__).resolve().parents[1],args.private_token)
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 1)

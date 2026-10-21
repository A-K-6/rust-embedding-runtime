#!/usr/bin/env python3
"""Collect dependency notices from the locked Cargo dependency sources."""
import json
from pathlib import Path
import subprocess

metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--locked', '--format-version', '1']))
sections = ['Third-party dependency licenses and notices\nGenerated from Cargo.lock. Project code is licensed separately.\n']
for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
    if package['id'] in metadata['workspace_members']:
        continue
    root = Path(package['manifest_path']).parent
    files = [p for p in root.iterdir() if p.is_file() and p.name.upper().startswith(('LICENSE', 'LICENCE', 'NOTICE', 'COPYING', 'COPYRIGHT'))]
    declared = package.get('license_file')
    if declared and (root / declared).is_file():
        files.append(root / declared)
    heading = f"\n{'=' * 72}\n{package['name']} {package['version']}\nLicense: {package.get('license') or 'See notice below'}\n"
    if not files:
        # Workspace crates sometimes omit root notices from their crate archive.
        repository = (package.get('repository') or '').rstrip('/')
        for sibling in metadata['packages']:
            if repository and (sibling.get('repository') or '').rstrip('/') == repository:
                sibling_root = Path(sibling['manifest_path']).parent
                files.extend(p for p in sibling_root.iterdir() if p.is_file() and p.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING', 'COPYRIGHT')))
        if not files and 'Apache-2.0' in (package.get('license') or ''):
            files = [Path('LICENSE')]
        if not files:
            raise SystemExit('Missing license source: ' + package['name'])
    sections.append(heading)
    for path in sorted(set(files)):
        sections.append('\n' + path.name + '\n' + path.read_text(errors='replace') + '\n')
Path('THIRD_PARTY_LICENSES.txt').write_text('\n'.join(line.rstrip() for line in ''.join(sections).splitlines()).rstrip() + '\n')

#!/usr/bin/env python3
"""Build the pinned AOCL-DLP static CPU library into a new, self-contained prefix."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

VERSION = '5.3.2'
REVISION = 'fe36da707e6f25e8b3e2b742dfa5957d4484fca5'
SOURCE_URL = f'https://codeload.github.com/amd/aocl-dlp/tar.gz/{REVISION}'
SOURCE_SHA256 = '694d4464e790159a57c069c123132f2a7e8145c3d1182f449b838aa11ba4032b'
SOURCE_ROOT = f'aocl-dlp-{REVISION}'
MAX_DOWNLOAD_BYTES = 32 * 1024**2
MAX_DOWNLOAD_SECONDS = 180
MAX_EXTRACT_BYTES = 128 * 1024**2
MAX_ARCHIVE_MEMBERS = 10_000
CHUNK_BYTES = 64 * 1024
OPTIONS = {
    'CMAKE_BUILD_TYPE': 'Release',
    'CMAKE_POSITION_INDEPENDENT_CODE': 'ON',
    'DLP_THREADING_MODEL': 'openmp',
    'BUILD_TESTING': 'OFF',
    'BUILD_BENCHMARKS': 'OFF',
    'BUILD_EXAMPLES': 'OFF',
    'BUILD_DOXYGEN': 'OFF',
    'BUILD_SPHINX': 'OFF',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            result.update(chunk)
    return result.hexdigest()


def download(destination):
    """Bound compressed bytes, socket stalls and total elapsed download time."""
    started = time.monotonic()
    request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'rust-embedding-runtime-aocl-builder'})
    result = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=30) as response, destination.open('xb') as output:
        declared = response.headers.get('Content-Length')
        if declared is not None and (not declared.isdecimal() or int(declared) > MAX_DOWNLOAD_BYTES):
            raise ValueError('Source archive exceeds the compressed download limit')
        while True:
            if time.monotonic()-started > MAX_DOWNLOAD_SECONDS:
                raise TimeoutError('Source archive download exceeded its time budget')
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError('Source archive exceeds the compressed download limit')
            result.update(chunk)
            output.write(chunk)
    if result.hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned AOCL-DLP source archive checksum mismatch')
    return total


def safe_extract(archive_path, destination):
    """Extract only verified ordinary files/directories, without tarfile.extract."""
    destination.mkdir()
    with tarfile.open(archive_path, 'r:gz') as archive:
        entries = []
        seen = set()
        total = 0
        for member in archive:
            if len(entries) >= MAX_ARCHIVE_MEMBERS:
                raise ValueError('Source archive contains too many members')
            name = PurePosixPath(member.name)
            if (not name.parts or name.is_absolute() or '..' in name.parts or
                    '\\' in member.name or '\0' in member.name or name.parts[0] != SOURCE_ROOT):
                raise ValueError('Unsafe path in AOCL-DLP source archive')
            if not (member.isdir() or member.isfile()):
                raise ValueError('Source archive contains links or special/device files')
            if name in seen:
                raise ValueError('Source archive contains duplicate paths')
            seen.add(name)
            if member.size < 0:
                raise ValueError('Source archive contains an invalid file size')
            total += member.size
            if total > MAX_EXTRACT_BYTES:
                raise ValueError('Source archive exceeds the expanded size limit')
            entries.append((member, destination.joinpath(*name.parts)))
        if not entries:
            raise ValueError('Source archive is empty')
        for member, path in entries:
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                path.chmod(0o755)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError('Source archive file could not be read')
            with source, path.open('xb') as output:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError('Source archive contains a truncated file')
                    output.write(chunk)
                    remaining -= len(chunk)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
    root = destination/SOURCE_ROOT
    if not root.is_dir():
        raise ValueError('Source archive has no expected root directory')
    return root


def executable(value):
    found = shutil.which(value)
    if found is None:
        raise ValueError(f'Required build executable is unavailable: {Path(value).name}')
    return str(Path(found).absolute())


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, timeout=30, **kwargs).strip()


def compiler_info(compiler, language, workspace):
    macros = capture([compiler, '-dM', '-E', '-x', language, '-'], input='')
    if '#define __GNUC__ ' not in macros or '#define __clang__ ' in macros:
        raise ValueError('AOCL-DLP build requires GCC C and C++ compilers')
    version = capture([compiler, '-dumpfullversion', '-dumpversion'])
    if re.fullmatch(r'\d+(?:\.\d+){0,2}', version) is None or int(version.split('.')[0]) < 13:
        raise ValueError('AOCL-DLP build requires GCC 13 or newer')
    target = capture([compiler, '-dumpmachine'])
    if re.fullmatch(r'x86_64-[A-Za-z0-9_.+-]+', target) is None or 'linux' not in target:
        raise ValueError('Compiler must target native x86_64 Linux')
    subprocess.run([compiler, '-march=znver4', '-x', language, '-c', '-o', str(workspace/f'{language}-probe.o'), '-'],
                   input='int aocl_compiler_probe(void) { return 0; }\n', text=True, check=True, timeout=30)
    return {'family': 'GCC', 'version': version, 'target': target, 'znver4_compile_probe': 'passed'}


def tool_version(tool, minimum=None):
    output = capture([tool, '--version'])
    match = re.search(r'(?m)^(?:cmake version )?(\d+\.\d+(?:\.\d+)?)\b', output)
    if match is None:
        raise ValueError(f'Cannot determine {Path(tool).name} version')
    version = match.group(1)
    if minimum is not None and tuple(map(int, version.split('.')[:2])) < minimum:
        raise ValueError('AOCL-DLP requires CMake 3.26 or newer')
    return version


def copy_archive(source, destination):
    if not source.is_file():
        raise ValueError(f'Required static archive is unavailable: {destination.name}')
    with source.open('rb') as stream:
        if stream.read(8) != b'!<arch>\n':
            raise ValueError(f'Expected a self-contained regular archive: {destination.name}')
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def copy_static_runtimes(cc, cxx, destination):
    for compiler, name in ((cxx, 'libstdc++.a'), (cc, 'libgomp.a'), (cc, 'libatomic.a')):
        resolved = capture([compiler, '-print-file-name='+name])
        source = Path(resolved)
        if resolved == name or not source.is_absolute():
            raise ValueError(f'Compiler did not resolve a static runtime archive: {name}')
        copy_archive(source, destination/name)


def publish(prefix, output):
    """Atomic same-filesystem publication that cannot replace an existing path."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('Atomic no-replace publication requires Linux renameat2 support')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(prefix), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, 'Cannot publish AOCL prefix without replacing an existing path')


def build(args):
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise ValueError('AOCL-DLP helper supports native Linux x86_64 only')
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError('Refusing to overwrite an existing AOCL output path')
    cc, cxx = executable(args.cc), executable(args.cxx)
    cmake, ninja = executable('cmake'), executable('ninja')
    cmake_version, ninja_version = tool_version(cmake, (3, 26)), tool_version(ninja)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.aocl-build-', dir=output.parent) as temporary:
        workspace = Path(temporary)
        compilers = {'c': compiler_info(cc, 'c', workspace), 'cxx': compiler_info(cxx, 'c++', workspace)}
        archive = workspace/'source.tar.gz'
        archive_bytes = download(archive)
        source = safe_extract(archive, workspace/'source')
        build_dir, prefix = workspace/'build', workspace/'prefix'
        environment = dict(os.environ, CC=cc, CXX=cxx)
        for key in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CMAKE_TOOLCHAIN_FILE'):
            environment.pop(key, None)
        command = [cmake, '-S', str(source), '-B', str(build_dir), '-G', 'Ninja',
                   '-DCMAKE_C_COMPILER='+cc, '-DCMAKE_CXX_COMPILER='+cxx,
                   '-DCMAKE_MAKE_PROGRAM='+ninja]
        command += ['-D'+key+'='+value for key, value in OPTIONS.items()]
        subprocess.run(command, env=environment, check=True, timeout=300)
        subprocess.run([cmake, '--build', str(build_dir), '--target', 'aocl-dlp_static', '--parallel', str(args.jobs)],
                       env=environment, check=True, timeout=3600)
        (prefix/'lib').mkdir(parents=True)
        for header in sorted((source/'include').rglob('*')):
            if header.is_file() and header.suffix in ('.h', '.hpp'):
                destination = prefix/'include'/header.relative_to(source/'include')
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(header, destination)
                destination.chmod(0o644)
        shutil.copyfile(build_dir/'include/aocl_dlp_config.h', prefix/'include/aocl_dlp_config.h')
        if not (prefix/'include/aocl_dlp.h').is_file():
            raise ValueError('Public AOCL-DLP headers were not produced')
        copy_archive(build_dir/'libaocl-dlp.a', prefix/'lib/libaocl-dlp.a')
        licenses = prefix/'share/licenses/aocl-dlp'
        licenses.mkdir(parents=True)
        for name in ('LICENSE', 'NOTICES'):
            shutil.copyfile(source/name, licenses/name)
            (licenses/name).chmod(0o644)
        if args.static_runtime:
            copy_static_runtimes(cc, cxx, prefix/'lib')
        artifacts = {path.relative_to(prefix).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
                     for path in sorted(prefix.rglob('*')) if path.is_file()}
        manifest = {'project': 'AOCL-DLP', 'version': VERSION, 'revision': REVISION,
                    'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256, 'source_archive_bytes': archive_bytes,
                    'platform': 'Linux x86_64', 'compilers': compilers,
                    'tools': {'cmake': cmake_version, 'ninja': ninja_version},
                    'build': {'generator': 'Ninja', 'target': 'aocl-dlp_static', 'jobs': args.jobs,
                              'options': OPTIONS, 'inherited_compiler_flags': False,
                              'global_architecture_override': False, 'compiler_static_runtimes': args.static_runtime},
                    'artifacts': artifacts}
        (prefix/'build-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        publish(prefix, output)
    print(json.dumps({'status': 'complete', 'project': 'AOCL-DLP', 'version': VERSION,
                      'revision': REVISION, 'source_sha256': SOURCE_SHA256}), flush=True)


def jobs(value):
    count = int(value)
    if not 1 <= count <= 64:
        raise argparse.ArgumentTypeError('--jobs must be between 1 and 64')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New output prefix; existing paths are never replaced')
    parser.add_argument('--jobs', type=jobs, default=2)
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'), help='GCC C compiler executable (13 or newer)')
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'), help='GCC C++ compiler executable (13 or newer)')
    parser.add_argument('--static-runtime', action='store_true', help='Also copy compiler-provided libstdc++.a, libgomp.a and libatomic.a')
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('Python 3.10 or newer is required')
    build(args)


if __name__ == '__main__':
    main()

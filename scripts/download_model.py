#!/usr/bin/env python3
"""Pinned, checked model download and text-only safetensors extraction."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import time
import urllib.request
import uuid

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
EXPECTED = {
    "model.safetensors": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "config.json": "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
}
SOURCE_BYTES = {"model.safetensors": 1488915288, "tokenizer.json": 32170510, "config.json": 4455}
DOWNLOAD_LIMITS = {"model.safetensors": 2 * 1024**3, "tokenizer.json": 64 * 1024**2, "config.json": 1024**2}
DOWNLOAD_CHUNK_BYTES = 4 * 1024**2


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def extract(source, dest):
    with source.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        n = struct.unpack("<Q", prefix)[0]
        if n > 4*1024*1024:
            raise ValueError("Invalid safetensors header")
        header = stream.read(n)
        if len(header) != n:
            raise ValueError("Truncated safetensors header")
        original = json.loads(header)
        base, offset = 8+n, 0
        selected = {"__metadata__": {"format": "pt", "source_revision": REVISION, "subset": "text-only"}}
        for name, item in original.items():
            if not name.startswith("language_model."):
                continue
            a,b = item["data_offsets"]
            if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b <= source.stat().st_size - base:
                raise ValueError("Invalid or truncated model tensor offsets")
            selected[name] = {"dtype": item["dtype"], "shape": item["shape"], "data_offsets": [offset,offset+b-a]}
            offset += b-a
        if len(selected) == 1:
            raise ValueError("No text model tensors found")
        header = json.dumps(selected,separators=(",",":")).encode()
        header += b" " * ((-len(header)) % 8)
        with dest.open("wb") as out:
            out.write(struct.pack("<Q",len(header)))
            out.write(header)
            for name in selected:
                if name == "__metadata__":
                    continue
                a,b = original[name]["data_offsets"]
                stream.seek(base+a)
                remaining = b-a
                while remaining:
                    chunk = stream.read(min(4*1024*1024,remaining))
                    if not chunk:
                        raise ValueError("Truncated model weights")
                    out.write(chunk)
                    remaining -= len(chunk)


def remaining_budget(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Download time budget exhausted")
    return remaining


def download(url, destination, max_bytes, deadline, socket_timeout):
    """A progress-checked deadline; blocking I/O is bounded by socket timeouts.

    This is not a killable wall-clock deadline. A blocking response operation
    can finish after the budget before the next progress check observes it.
    """
    timeout = min(socket_timeout, 120, remaining_budget(deadline))
    with urllib.request.urlopen(url, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None:
            text = str(length).strip()
            if not text.isdigit():
                raise ValueError("Invalid Content-Length")
            if int(text) > max_bytes:
                raise ValueError("Download Content-Length exceeds file bound")
        received = 0
        read = getattr(response, "read1", response.read)
        with destination.open("wb") as out:
            while True:
                remaining_budget(deadline)
                block = read(min(DOWNLOAD_CHUNK_BYTES, max_bytes - received + 1))
                remaining_budget(deadline)
                if not block:
                    break
                if len(block) > max_bytes - received:
                    raise ValueError("Download stream exceeds file bound")
                out.write(block)
                received += len(block)
        return received


def validate_output_targets(output):
    for name in (*EXPECTED, "manifest.json"):
        target = output / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError("Refusing a symlink or non-file model target")


def publish(stage, output):
    """Promote checked files with the manifest last, rolling back failures."""
    names = [*EXPECTED, "manifest.json"]
    validate_output_targets(output)
    previous = stage / "previous"
    previous.mkdir(mode=0o700)
    moved, installed = [], []
    try:
        for name in names:
            target = output / name
            if target.exists():
                if name == "manifest.json":
                    # Retain the old manifest until its final atomic replace.
                    shutil.copy2(target, previous / name)
                else:
                    target.replace(previous / name)
                moved.append(name)
            (stage / name).replace(target)
            installed.append(name)
    except BaseException as failure:
        # Only this call's promoted files/backups are addressed. Other files in
        # the destination are preserved, including an existing valid manifest.
        rollback_failed = False
        for name in reversed(installed):
            try:
                (output / name).unlink(missing_ok=True)
            except OSError:
                rollback_failed = True
        for name in reversed(moved):
            try:
                (previous / name).replace(output / name)
            except OSError:
                rollback_failed = True
        if rollback_failed:
            # Keep recoverable originals if the filesystem also refuses their
            # restoration; the TemporaryDirectory must not delete backups.
            recovery = output / (".embedding-runtime-model-recovery-" + uuid.uuid4().hex)
            stage.replace(recovery)
            raise RuntimeError("Model publication rollback failed; recoverable files retained at " + str(recovery)) from failure
        raise


def acquire(output, download_temp_dir=None, download_budget_seconds=1800, socket_timeout_seconds=120):
    if not math.isfinite(download_budget_seconds) or not 0 < download_budget_seconds <= 86400:
        raise ValueError("Download budget must be positive and at most 86400 seconds")
    if not math.isfinite(socket_timeout_seconds) or not 0 < socket_timeout_seconds <= 120:
        raise ValueError("Socket timeout must be positive and at most 120 seconds")
    output = output.expanduser().absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Refusing a symlink or non-directory model output")
    output = output.resolve()
    validate_output_targets(output)
    if download_temp_dir is not None:
        download_temp_dir = download_temp_dir.expanduser().resolve()
        if not download_temp_dir.is_dir():
            raise ValueError("Download temporary directory must already exist")
    created_output = not output.exists()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    deadline = time.monotonic() + download_budget_seconds
    downloaded = {}
    try:
        # The full public source weights can use /dev/shm, while checked output
        # staging stays on the output filesystem for atomic renames.
        with tempfile.TemporaryDirectory(prefix="embeddinggemma2-download-", dir=download_temp_dir) as temp, tempfile.TemporaryDirectory(prefix=".embedding-runtime-model-stage-", dir=output) as stage:
            temp, stage = Path(temp), Path(stage)
            for name, expected in EXPECTED.items():
                destination = temp / name
                url = f"https://huggingface.co/google/embeddinggemma-2/resolve/{REVISION}/{name}"
                downloaded[name] = download(url, destination, DOWNLOAD_LIMITS[name], deadline, socket_timeout_seconds)
                if sha(destination) != expected:
                    raise ValueError("Checksum mismatch: " + name)
                remaining_budget(deadline)
                if name == "model.safetensors":
                    extract(destination, stage / name)
                else:
                    shutil.copyfile(destination, stage / name)
                (stage / name).chmod(0o600)
            manifest = {"revision": REVISION, "subset": "text-only", "files": {}}
            for name in EXPECTED:
                destination = stage / name
                manifest["files"][name] = {"sha256": sha(destination), "bytes": destination.stat().st_size}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
            (stage / "manifest.json").chmod(0o600)
            remaining_budget(deadline)
            publish(stage, output)
    except BaseException:
        if created_output:
            try:
                output.rmdir()
            except OSError:
                pass
        raise
    return {"model_dir": str(output), "revision": REVISION, "bytes": sum(x["bytes"] for x in manifest["files"].values()),
            "downloaded_bytes": downloaded, "pinned_source_bytes": SOURCE_BYTES,
            "download_budget_seconds": download_budget_seconds, "socket_timeout_seconds": socket_timeout_seconds,
            "deadline_semantics": "Progress checked between blocking response operations; socket timeouts bound I/O inactivity, not a strict total wall-clock deadline."}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent.parent / "model")
    parser.add_argument("--download-temp-dir", type=Path, help="Existing filesystem for full downloaded weights, e.g. /dev/shm")
    parser.add_argument("--download-budget-seconds", type=float, default=1800, help="Progress-checked total acquisition budget; not a strict wall-clock deadline")
    parser.add_argument("--socket-timeout-seconds", type=float, default=120, help="I/O inactivity timeout, at most 120 seconds")
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.output, args.download_temp_dir, args.download_budget_seconds, args.socket_timeout_seconds)))


if __name__ == "__main__":
    main()

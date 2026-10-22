"""Verify the checked-in historical evidence using only the standard library."""

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = json.loads((ROOT / "results/sources.json").read_text())
    for record in records:
        if sha256(ROOT / record["destination"]) != record["sha256"]:
            raise ValueError(f"Historical result changed: {record['destination']}")
    archive = ROOT / "archive/historical-source.tar.gz"
    index = json.loads((ROOT / "archive/index.json").read_text())
    if sha256(archive) != index["archive_sha256"]:
        raise ValueError("Historical source archive changed")
    expected = {item["path"]: item["sha256"] for item in index["files"]}
    with tarfile.open(archive) as handle:
        if {m.name for m in handle.getmembers()} != set(expected):
            raise ValueError("Archive member coverage changed")
        for member in handle.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unexpected archive member")
            if hashlib.sha256(handle.extractfile(member).read()).hexdigest() != expected[member.name]:
                raise ValueError(f"Archived source changed: {member.name}")
    print(f"Verified {len(records)} historical result files and {len(expected)} archive members")


if __name__ == "__main__":
    main()

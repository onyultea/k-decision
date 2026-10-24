"""JSONL 입출력 — 허용 디렉터리 화이트리스트 검증 내장.

모든 JSONL 읽기/쓰기는 이 모듈을 통해서만 수행한다.
정책: 경로는 'data/<파일명>' 또는 'reports/<파일명>' 형식의 화이트리스트 정규식만
허용(파일명은 [A-Za-z0-9._-]), 절대 루트는 프로젝트 루트(src의 상위)로 고정.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAFE_REL = re.compile(r"^(?:data|reports)/[A-Za-z0-9._-]+$")


def _safe_data_path(path) -> Path:
    s = str(path)
    if not _SAFE_REL.match(s):
        raise ValueError("허용 경로 형식: data/<파일명> 또는 reports/<파일명>")
    return _PROJECT_ROOT / s


def load_jsonl(path):
    p = _safe_data_path(path)
    text = p.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def dump_jsonl(recs, path):
    p = _safe_data_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
        encoding="utf-8",
    )

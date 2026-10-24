"""Python 3.10 호환 심 — hashlib.file_digest (3.11+ API) 제공.

Decision Index 킷의 chessbench 빌더가 hashlib.file_digest 를 사용하는데
서버 venv(3.10)에는 해당 API가 없어 AttributeError 로 죽는다.
이 파일을 venv site-packages 에 sitecustomize.py 로 두면 모든 프로세스에 적용된다.
"""
import hashlib

if not hasattr(hashlib, "file_digest"):  # pragma: no cover - 3.11+ 에서는 미실행
    def _file_digest(fileobj, algorithm, /, *, _bufsize=2 ** 18):
        h = hashlib.new(algorithm)
        if hasattr(fileobj, "readinto"):
            buf = bytearray(_bufsize)
            view = memoryview(buf)
            while True:
                n = fileobj.readinto(buf)
                if not n:
                    break
                h.update(view[:n])
        else:
            while True:
                block = fileobj.read(_bufsize)
                if not block:
                    break
                h.update(block)
        return h

    hashlib.file_digest = _file_digest

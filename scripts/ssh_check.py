"""SSH 연결 검증 + 공개키 자동 설치 (ssh-copy-id 대체, 일반화).

사용법:
  1. secrets/server.txt 에 host/port/user/password 작성 (gitignore 대상)
  2. python scripts/ssh_check.py          → 연결 + GPU 확인
  3. python scripts/ssh_check.py --install-key  → 공개키 등록
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import paramiko

CREDS = Path(__file__).resolve().parent.parent / "secrets" / "server.txt"


def read_creds():
    raw = CREDS.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    vals = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    missing = [k for k in ("host", "port", "user", "password") if not vals.get(k)]
    if missing:
        sys.exit(f"secrets/server.txt 에 {missing} 값이 비어 있음 — 채워 주세요.")
    return vals


def connect(creds):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        creds["host"],
        port=int(creds["port"]),
        username=creds["user"],
        password=creds["password"],
        timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    return c


def run(c, cmd):
    _, out, err = c.exec_command(cmd, timeout=60)
    o, e = out.read().decode(errors="replace").strip(), err.read().decode(errors="replace").strip()
    return o, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-key", action="store_true",
                    help="이 PC의 공개키(~/.ssh/id_ed25519.pub)를 원격에 등록")
    args = ap.parse_args()

    creds = read_creds()
    print(f"접속 시도: {creds['user']}@{creds['host']}:{creds['port']}")
    c = connect(creds)
    print("연결 성공!")

    o, _ = run(c, "hostname && uname -a")
    print(f"호스트: {o}")

    o, _ = run(c, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1")
    print(f"GPU:\n{o}")

    if args.install_key:
        pub = Path.home() / ".ssh" / "id_ed25519.pub"
        if not pub.exists():
            sys.exit("공개키가 없음: 먼저 ssh-keygen -t ed25519 실행 필요")
        key = pub.read_bytes().decode(errors="replace").strip()
        cmd = (
            'mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
            f'grep -qF "{key}" ~/.ssh/authorized_keys 2>/dev/null || '
            f'echo "{key}" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; echo KEY_OK'
        )
        o, e = run(c, cmd)
        print("공개키 등록:", o if "KEY_OK" in o else f"실패 {e}")
    c.close()


if __name__ == "__main__":
    main()

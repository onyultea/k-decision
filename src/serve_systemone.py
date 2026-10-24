"""K-Decision `/v1/systemone` 서버 — Decision Index 킷 연동용.

Decision Index reproduction kit(github.com/apolinario/decision-index)의
HttpSystemOne 엔진이 호출하는 POST /v1/systemone 를 구현한다.
요청/응답 계약은 decision_index.engines.base.validate 를 그대로 만족:

  요청  {"model": ..., "state": str, "questions": {qid: {
           "type": "choice"|"noul", "instructions": str,
           "criteria": {key: 설명}   (choice만) }}}
  응답  {"model": ..., "answers": {qid:
           choice → {"type":"choice","choice":key,"probabilities":{key:p,...}}
           noul   → {"type":"noul","noul": P(yes)}}}

용량 초과(옵션>8, 컨텍스트 초과)는 422 + CAPACITY_MARKERS 문구로 거부한다.
실행: python serve_systemone.py --merged <병합모델경로> --port 8111
"""
from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer

import common as C

CAPACITY_MARKERS = (
    "options per choice", "a choice needs at least two options",
    "a score takes 2 to 10 levels", "the canvas holds",
    "maximum context length", "maximum model length",
    "longer than the maximum model length", "context window", "too many tokens",
)
MAX_OPTIONS = 26         # 체스 합법수술 등 대응 (A..Z)
MAX_STATE_CHARS = 6000   # 초과 시 용량 거부 (초장문 로짓 OOM 방지)

_model = None
_tok = None
_marker_id = None
_meta = None
_lock = threading.Lock()


class Capacity(Exception):
    pass


def load(merged_dir: str, int8: bool = False):
    global _model, _tok, _marker_id, _meta
    from train import load_model

    _tok = AutoTokenizer.from_pretrained(merged_dir)
    marker = C.resolve_marker(_tok)
    _marker_id = _tok.encode(marker, add_special_tokens=False)[0]
    _model = load_model(merged_dir, for_training=False,
                        dtype=torch.bfloat16, load_in_8bit=int8)
    if not int8:
        _model = _model.cuda()
    _model.eval()
    import json as _json
    from pathlib import Path

    meta_path = Path(merged_dir) / "kdecision_meta.json"
    _meta = _json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    print(f"[serve] 모델 로드 완료: {merged_dir} (marker={marker!r})")


def wire_to_record(state, questions: dict) -> dict:
    # state 가 문자열이 아니면(딕셔너리 등) JSON 으로 직렬화해 프롬프트에 문자열만 흐르게 한다
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    # 마커 문자 이스케이핑: state 내 '□'가 마커 카운팅을 오염시키지 않게 치환 (Kev 교훈)
    state = state.replace("□", "▪").replace("@", "＠")
    qs, answers_meta = [], []
    for qid, q in questions.items():
        if q["type"] == "choice":
            criteria = q.get("criteria")
            if not isinstance(criteria, dict):
                raise Capacity(f"unsupported: a choice needs criteria; {type(criteria)}")
            keys = list(criteria)
            if len(keys) > MAX_OPTIONS:
                raise Capacity(f"unsupported: more than {MAX_OPTIONS} options per choice")
            if len(keys) < 2:
                raise Capacity("unsupported: a choice needs at least two options")
            # 옵션 표기: 설명이 자명하면 설명 사용, 아니면 키 자체
            opts = [str(criteria[k]) or str(k) for k in keys]
            qs.append({"id": qid, "type": "choice", "text": q.get("instructions", ""),
                       "options": opts})
            answers_meta.append(("choice", keys))
        elif q["type"] == "noul":
            qs.append({"id": qid, "type": "noul", "text": q.get("instructions", "")})
            answers_meta.append(("noul", None))
        else:
            raise Capacity(f"unsupported: a score takes 2 to 10 levels ({q['type']} 미지원)")
    return {"state": state, "questions": qs}, answers_meta


@torch.no_grad()
def decide(state: str, questions: dict) -> dict:
    rec, answers_meta = wire_to_record(state, questions)
    if len(rec["state"]) > MAX_STATE_CHARS:
        raise Capacity("unsupported: state longer than the maximum model length")

    msgs = [
        {"role": "user", "content": C.build_user_content(rec)},
        {"role": "assistant", "content": C.build_assistant_skeleton(rec)},
    ]
    text = _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = _tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    try:
        with _lock:
            logits = _model(input_ids=ids).logits[0].float()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise Capacity("unsupported: prompt longer than the maximum model length")

    marker_pos = (ids[0] == _marker_id).nonzero(as_tuple=True)[0].tolist()
    assert len(marker_pos) == len(rec["questions"]), "마커/질문 수 불일치"

    out = {}
    for q, p, (kind, keys) in zip(rec["questions"], marker_pos, answers_meta):
        syms = C.CHOICE_SYMBOLS[: len(q["options"])] if q["type"] == "choice" \
            else C.NOUL_SYMBOLS
        cand_ids = C.symbol_ids(_tok, syms)
        probs = torch.softmax(logits[p - 1, cand_ids], dim=-1)
        if kind == "choice":
            pmap = {k: float(probs[i]) for i, k in enumerate(keys)}
            top = max(pmap, key=pmap.get)
            out[q["id"]] = {"type": "choice", "choice": top, "probabilities": pmap}
        else:  # noul: P(Y)
            yi = syms.index("Y")
            out[q["id"]] = {"type": "noul", "noul": float(probs[yi])}
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in ("/v1/systemone", "/v1/decisions", "/v1/jev"):
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n))
            answers = decide(payload["state"], payload["questions"])
        except Capacity as e:
            return self._send(422, {"error": str(e)})
        except Exception as e:  # 킷 규칙: 기타 오류는 500 → 'error' 처리
            import traceback
            traceback.print_exc()  # serve.log 에 트레이스백 기록
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(200, {"model": payload.get("model", "k-decision"), "answers": answers})

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):  # 로그 정숙화
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--int8", action="store_true",
                    help="9B 병합 모델 int8 서빙 (OOM 제거, 헤드룸 확보)")
    args = ap.parse_args()
    load(args.merged, int8=args.int8)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] http://127.0.0.1:{args.port} 대기 (/v1/systemone)")
    srv.serve_forever()


if __name__ == "__main__":
    main()

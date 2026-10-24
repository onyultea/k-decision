"""YURIS-2: 투-타워 어모타이즈드 결정 네트워크.

상태 타워(긴 state 1회 인코딩 → 캐시) + 옵션 타워(질문×옵션, 저렴) + FiLM 퓨전.
메타인지 헤드: 주석자 합의 여부 예측 (애매 문제 탐지 → 에스컬레이션).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class YURIS2(nn.Module):
    def __init__(self, backbone: str = "BAAI/bge-m3", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.ln = nn.LayerNorm(h)
        # FiLM: h_state → (γ, β) 로 h_opt를 변조
        self.film = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 2 * h))
        self.trunk = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout))
        self.score_head = nn.Linear(h, 1)
        self.noul_head = nn.Linear(h, 1)
        self.agreement_head = nn.Linear(h, 1)
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.xavier_uniform_(self.noul_head.weight)
        nn.init.xavier_uniform_(self.agreement_head.weight)

    def encode_state(self, input_ids, attention_mask):
        """상태 타워: 긴 state → h_state (캐시 가능)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    def encode_option(self, input_ids, attention_mask):
        """옵션 타워: 질문+옵션 → h_opt (저렴)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(out.last_hidden_state, attention_mask)

    @staticmethod
    def _pool(h, mask):
        """CLS 토큰 풀링 — ModernBERT/BERT 계열 표준."""
        return h[:, 0]

    def fuse(self, h_state, h_opt):
        """FiLM 퓨전: 상태 표현으로 옵션 표현을 변조."""
        gamma, beta = self.film(h_state).chunk(2, dim=-1)
        fused = gamma * h_opt + beta + h_opt
        return self.ln(fused)

    def score(self, fused):
        return self.score_head(fused).squeeze(-1)

    def agreement(self, fused):
        return self.agreement_head(fused).squeeze(-1)

    def forward(self, state_ids, state_mask, opt_ids, opt_mask):
        """학습용 전체 forward (캐싱 없이 한 번에)."""
        h_state = self.encode_state(state_ids, state_mask)
        h_opt = self.encode_option(opt_ids, opt_mask)
        fused = self.fuse(h_state, h_opt)
        return self.score(fused), self.agreement(fused)


def build(backbone: str = "BAAI/bge-m3"):
    return YURIS2(backbone)

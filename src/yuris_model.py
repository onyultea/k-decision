"""YURIS-Decision 모델 — 멀티프리미티브 인코더 + 결정 헤드.

백본(멀티링구얼 인코더) + 프리미티브별 스코어링 헤드.
인과 LLM 디시전 모델(교사)의 확률 분포를 증류하여 학습.

입력 (질문×옵션 쌍당 1 시퀀스):
  [CLS] state </s> question </s> option_i
출력: option_i의 스코어 → 질문 내 옵션 softmax/sigmoid
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class YURISDecision(nn.Module):
    def __init__(self, backbone: str = "BAAI/bge-m3", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # 프리미티브별 스코어링 헤드 (공유 인코더 + 독립 헤드)
        self.choice_head = nn.Linear(hidden, 1)
        self.noul_head = nn.Linear(hidden, 1)
        self.score_head = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.choice_head.weight)
        nn.init.xavier_uniform_(self.noul_head.weight)
        nn.init.xavier_uniform_(self.score_head.weight)

    def forward(self, input_ids, attention_mask):
        """입력 시퀀스들의 옵션 표현 → 프리미티브별 스코어."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(2))
        pooled = h.gather(1, idx).squeeze(1)
        pooled = self.dropout(pooled)
        return (self.choice_head(pooled).squeeze(-1),
                self.noul_head(pooled).squeeze(-1),
                self.score_head(pooled).squeeze(-1))


def build(backbone: str = "BAAI/bge-m3", dropout: float = 0.1):
    return YURISDecision(backbone, dropout)

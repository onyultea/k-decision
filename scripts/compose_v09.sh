#!/bin/bash
# v0.9 믹스: 격전지(소프트라벨+전길이) 6배 + 규정 + 한국생활 + 증류 + tasksource 축소
set -e
cd /home/kitri/k-decision
export HF_TOKEN=$(cat /home/kitri/.hf-token)
/home/kitri/venvs/di/bin/python src/prep_typeddec.py --soft-train
/home/kitri/venvs/di/bin/python - << "EOF"
import json, random, sys
sys.path.insert(0, "src")
def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
klife = load("data/klife_v02_train.jsonl")
synth = load("data/synth_v01_train.jsonl")
tdc = load("data/typeddec_train.jsonl")
ts = load("data/tasksource_train.jsonl")
cd = load("data/distill_train.jsonl")
td = load("data/tooldec_train.jsonl")
reg = load("data/regul_v08_train.jsonl")
random.Random(5).shuffle(ts)
mix = klife[:20000] + synth[:3000] + tdc * 6 + ts[:15000] + cd + td + reg * 4
random.Random(2).shuffle(mix)
with open("data/mix_v09_train.jsonl", "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n_soft = sum(1 for r in mix for q in r["questions"] if "soft" in q)
print("v0.9 믹스:", len(mix), "레코드 / 소프트라벨 질문", n_soft)
EOF
echo COMPOSE9_DONE

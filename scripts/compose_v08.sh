#!/bin/bash
# v0.8 믹스: 격전지 5배 업샘플 + 규정 준거 + 한국생활 + 증류(소프트라벨) + tasksource
set -e
cd /home/kitri/k-decision
/home/kitri/venvs/di/bin/python - << "EOF"
import json, random, sys
sys.path.insert(0, "src")
def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
klife = load("data/klife_v02_train.jsonl")
synth = load("data/synth_v01_train.jsonl")
gen = load("data/generic_v02_train.jsonl")
tdc = load("data/typeddec_train.jsonl")
ts = load("data/tasksource_train.jsonl")
cd = load("data/distill_train.jsonl")
td = load("data/tooldec_train.jsonl")
reg = load("data/regul_v08_train.jsonl")
random.Random(1).shuffle(gen); random.Random(5).shuffle(ts)
mix = klife + synth + gen[:8000] + tdc * 5 + ts[:30000] + cd + td + reg * 4
random.Random(2).shuffle(mix)
with open("data/mix_v08_train.jsonl", "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("v0.8 믹스:", len(klife), "+", len(synth), "+ 8000 +", len(tdc)*5, "+", min(30000,len(ts)), "+", len(cd), "+", len(td), "+", len(reg)*4, "=", len(mix))
EOF
echo COMPOSE8_DONE

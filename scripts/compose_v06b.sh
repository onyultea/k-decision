#!/bin/bash
# v0.6b 믹스 재구성: 기존 78.2k + 클로드증류(한국어중심) + 도구결정
set -e
cd /home/kitri/k-decision
/home/kitri/venvs/di/bin/python src/convert_extra_pools.py --ko 8000 --other 3000 --tools 5000
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
random.Random(1).shuffle(gen); random.Random(5).shuffle(ts)
mix = klife + synth + gen[:8000] + tdc + ts[:30000] + cd + td
random.Random(2).shuffle(mix)
with open("data/mix_v06_train.jsonl", "w", encoding="utf-8") as f:
    for r in mix:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("v0.6b 믹스:", len(klife), "+", len(synth), "+ 8000 +", len(tdc), "+", min(30000,len(ts)), "+", len(cd), "+", len(td), "=", len(mix))
EOF
/home/kitri/venvs/di/bin/python src/prep_kjdi.py --sample 3000
echo COMPOSE_DONE

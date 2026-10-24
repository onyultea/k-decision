#!/bin/bash
# K-Decision 체인 v9 — v12: 2단계 학습 (leader 2개의 "general-then-finetune" 구조 재현)
# 단계1: 일반 디시전 데이터(tasksource+증류+도구+일반) 1에포크
# 단계2: 단계1 어댑터에서 격전지(소프트)×8 + 한국생활 + 규정 집중 2에포크
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/6] 믹스 2종 구성"
/home/kitri/venvs/di/bin/python - << "EOF"
import json, random, sys
sys.path.insert(0, "src")
def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
ts = load("data/tasksource_train.jsonl")
cd = load("data/distill_train.jsonl")
td = load("data/tooldec_train.jsonl")
gen = load("data/generic_v02_train.jsonl")
klife = load("data/klife_v02_train.jsonl")
synth = load("data/synth_v01_train.jsonl")
tdc = load("data/typeddec_train.jsonl")
reg = load("data/regul_v08_train.jsonl")
random.Random(5).shuffle(ts); random.Random(1).shuffle(gen)
# 단계1: 일반 디시전 ~49k
s1 = ts[:30000] + cd + td + gen[:8000]
random.Random(2).shuffle(s1)
with open("data/mix_v12_stage1.jsonl", "w", encoding="utf-8") as f:
    for r in s1: f.write(json.dumps(r, ensure_ascii=False) + "\n")
# 단계2: 격전지 소프트×8 + 한국 + 규정 ~57k
s2 = tdc * 8 + klife[:20000] + synth[:3000] + reg * 4
random.Random(3).shuffle(s2)
with open("data/mix_v12_stage2.jsonl", "w", encoding="utf-8") as f:
    for r in s2: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"stage1={len(s1)} stage2={len(s2)}")
EOF

echo "[2/6] 스모크"
head -n 300 data/mix_v12_stage1.jsonl > data/smoke_mix.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_mix.jsonl \
  --out /tmp/kd-smoke12 --max_steps 3 --bs 1 --accum 1 --lr 7e-5 --max_len 1088 --int8 --bf16 > smoke_v12.log 2>&1
[ -d /tmp/kd-smoke12/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/kd-smoke12 data/smoke_mix.jsonl

echo "[3/6] 단계1: 일반 디시전 학습 (9B int8, 1에포크)"
rm -f train_v12_s1.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v12_stage1.jsonl \
  --out models/kd-9b-v12-stage1 --epochs 1 --bs 1 --accum 16 --lr 1e-4 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v12_s1.log 2>&1
[ -d models/kd-9b-v12-stage1/adapter ] || { echo "단계1 실패"; exit 1; }

echo "[4/6] 단계2: 격전지 집중 학습 (단계1 어댑터에서, 2에포크)"
rm -f train_v12.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v12_stage2.jsonl \
  --init_adapter models/kd-9b-v12-stage1/adapter \
  --out models/kd-9b-v12-3090 --epochs 2 --bs 1 --accum 16 --lr 3e-5 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v12.log 2>&1
[ -d models/kd-9b-v12-3090/adapter ] || { echo "단계2 실패"; exit 1; }

echo "[5/6] 판정 평가"
MB=Qwen/Qwen3.5-9B; MM=models/kd-9b-v12-3090/kdecision_meta.json; MA=models/kd-9b-v12-3090/adapter
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag kd_v12_typeddec > eval_v12_typeddec.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 1 --int8 --out reports --tag kd_v12_klife > eval_v12_klife.log 2>&1
V11=$(python3 -c "import json;print(json.load(open('reports/kd_v11_typeddec.json'))['raw']['acc'])")
V12=$(python3 -c "import json;print(json.load(open('reports/kd_v12_typeddec.json'))['raw']['acc'])")
V12KL=$(python3 -c "import json;print(json.load(open('reports/kd_v12_klife.json'))['raw']['acc'])")
echo "격전지: v11=$V11 vs v12=$V12 (1위=0.826) / klife=$V12KL"

echo "[6/6] 판정"
if python3 -c "exit(0 if $V12 > 0.818 else 1)"; then
  echo "v12 개선 — 가중 앙상블(v12+v11) 최종 탐색 후 제출 갱신 대상"
fi
echo "CHAIN9_ALL_DONE v12=$V12 $(date)"

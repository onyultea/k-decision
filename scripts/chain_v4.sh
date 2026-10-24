#!/bin/bash
# K-Decision 체인 v4 — v0.7 (소프트라벨 CE + 격전지 업샘플) Darwin 역전 시도
# 체인 v3 완료 대기 → kjdi 보완측정(bs1) → v0.7 학습 → 6종 평가
# → (typeddec 역전 & klife 유지) 시 병합·서빙·DI 재측정
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/7] 체인 v3(DI v0.6) 완료 대기..."
while ! grep -q "CHAIN3_ALL_DONE" chain_v3.log 2>/dev/null; do sleep 300; done
echo "체인 v3 완료: $(date)"

echo "[2/7] v0.6 잔여서빙 정리 + kjdi 보완측정(배치1)"
pkill -f serve_systemone 2>/dev/null; sleep 5
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-4B --adapter models/kd-v06-3090/adapter --meta models/kd-v06-3090/kdecision_meta.json --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v06_kjdi > eval_v06_kjdi2.log 2>&1
echo "kjdi v0.6: $(grep -aoE '\"acc\": [0-9.]+' reports/kd_v06_kjdi.json 2>/dev/null | head -1)"

echo "[3/7] v0.7 믹스 구성"
bash scripts/compose_v07.sh > compose_v07.log 2>&1 || { echo "믹스 구성 실패"; exit 1; }

echo "[4/7] v0.7 스모크(0.8B 3스텝, 소프트라벨 경로 검증) 후 본학습"
head -n 300 data/mix_v07_train.jsonl > data/smoke_mix.jsonl
nice -n 10 $KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_mix.jsonl \
  --out /tmp/kd-smoke --max_steps 3 --bs 1 --accum 1 --lr 7e-5 > smoke_v07.log 2>&1
[ -d /tmp/kd-smoke/adapter ] || { echo "스모크 실패 — 소프트라벨 경로 오류, 학습 중단"; exit 1; }
rm -rf /tmp/kd-smoke data/smoke_mix.jsonl
rm -f train_v07.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v07_train.jsonl \
  --out models/kd-v07-3090 --epochs 1.5 --bs 2 --accum 8 --lr 7e-5 \
  > train_v07.log 2>&1
[ -d models/kd-v07-3090/adapter ] || { echo "v0.7 학습 실패"; exit 1; }

echo "[5/7] v0.7 6종 평가"
MB=Qwen/Qwen3.5-4B; MM=models/kd-v07-3090/kdecision_meta.json; MA=models/kd-v07-3090/adapter
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v07_klife > eval_v07_klife.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v07_synth > eval_v07_synth.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v07_kmmlu > eval_v07_kmmlu.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag kd_v07_kobest > eval_v07_kobest.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag kd_v07_typeddec > eval_v07_typeddec.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v07_kjdi > eval_v07_kjdi.log 2>&1

V6TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v06_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V7TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v07_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V7KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v07_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "격전지: v0.6=$V6TDC vs v0.7=$V7TDC / klife v0.7=$V7KLIFE"

echo "[6/7] 승패 판정"
if python3 -c "exit(0 if $V7TDC > $V6TDC and $V7KLIFE >= 0.995 else 1)"; then
  echo "v0.7 역전 성공 — 병합·서빙·DI 재측정"
  $KD5 src/merge_adapter.py --base $MB --adapter $MA --marker_meta $MM --out models/kd-v07-merged > merge_v07.log 2>&1
  nohup $KD5 src/serve_systemone.py --merged models/kd-v07-merged --port 8111 >> serve.log 2>&1 &
  sleep 30
  curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
  nohup $DI -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8111 --option model=k-decision-v0.7 \
    --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-v0.7 > di_pipeline_v07.log 2>&1 &
  while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
else
  echo "v0.7 미역전 — v0.6 유지"
fi
echo "CHAIN4_ALL_DONE $(date)"

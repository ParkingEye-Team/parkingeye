#!/bin/bash
# 多段并行下载器: parallel_download.sh <url> <output> <n_parts>
set -e
URL="$1"; OUT="$2"; N="${3:-24}"
SIZE=$(curl -sI "$URL" | grep -i content-length | tail -1 | tr -d '\r' | awk '{print $2}')
echo "size=$SIZE parts=$N"
CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  S=$((i * CHUNK)); E=$((S + CHUNK - 1))
  [ $E -ge $SIZE ] && E=$((SIZE - 1))
  (
    for try in $(seq 1 200); do
      have=0; [ -f "$OUT.part$i" ] && have=$(stat -c %s "$OUT.part$i")
      want=$((E - S + 1))
      [ "$have" -ge "$want" ] && break
      # -y 30 -Y 10240: 30 秒内速度低于 10KB/s 则断开；--max-time 防挂死；断点续传
      curl -s -r $((S + have))-$E --max-time 300 -y 30 -Y 10240 --connect-timeout 15 "$URL" >> "$OUT.part$i" || sleep 2
    done
  ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.part$i"; done) > "$OUT"
rm -f "$OUT".part*
echo "done: $(stat -c %s "$OUT") bytes -> $OUT"

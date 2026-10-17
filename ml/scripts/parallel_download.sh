#!/bin/bash
# 多段并行下载器: parallel_download.sh <url> <output> <n_parts>
# 修复（doc/11 P2-13）：校验 Content-Length、Range(206) 支持、最终大小。
set -euo pipefail
URL="${1:?用法: parallel_download.sh <url> <output> [n_parts]}"
OUT="${2:?缺少输出路径}"
N="${3:-24}"

HEADERS=$(curl -sIL "$URL" | tr -d '\r' || true)
SIZE=$(echo "$HEADERS" | grep -i '^content-length:' | tail -1 | awk '{print $2}' || true)
echo "size=$SIZE parts=$N"

if ! [[ "$SIZE" =~ ^[0-9]+$ ]] || [ "$SIZE" -le 0 ]; then
  echo "错误：无法获取有效的 Content-Length（SIZE='${SIZE:-}'），终止。" >&2
  exit 1
fi

# 服务器必须支持 Range（返回 206）；否则多段会拼坏文件
STATUS=$(curl -s -o /dev/null -w '%{http_code}' -r 0-0 "$URL" || true)
if [ "$STATUS" != "206" ]; then
  echo "警告：Range 请求返回 HTTP $STATUS（非 206），改用单线程断点续传。" >&2
  curl -fL --retry 20 --retry-delay 2 -C - -o "$OUT" "$URL"
  echo "done: $(stat -c %s "$OUT") bytes -> $OUT"
  exit 0
fi

CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  S=$((i * CHUNK)); E=$((S + CHUNK - 1))
  [ $E -ge $SIZE ] && E=$((SIZE - 1))
  (
    for try in $(seq 1 200); do
      have=0
      [ -f "$OUT.part$i" ] && have=$(stat -c %s "$OUT.part$i")
      want=$((E - S + 1))
      [ "$have" -ge "$want" ] && break
      # -f: HTTP 错误直接失败；-y/-Y: 低速断开保护；--max-time 防挂死；断点续传
      curl -fs -r $((S + have))-$E --max-time 300 -y 30 -Y 10240 \
           --connect-timeout 15 "$URL" >> "$OUT.part$i" || sleep 2
    done
  ) &
done
wait

# 拼接（避免空格路径被 cat $(...) 拆开）
: > "$OUT"
for i in $(seq 0 $((N-1))); do
  cat "$OUT.part$i" >> "$OUT"
done
rm -f "$OUT".part*

ACTUAL=$(stat -c %s "$OUT")
if [ "$ACTUAL" -ne "$SIZE" ]; then
  echo "错误：下载大小 $ACTUAL != 期望 $SIZE，文件可能不完整。" >&2
  exit 1
fi
echo "done: $ACTUAL bytes -> $OUT"

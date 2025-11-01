# KW="your_keyword"   # 여기에 찾을 키워드

# 1) 미리 보기(전체 커맨드 라인과 PID)
pgrep -af -- "$KW" || { echo "No matches."; exit 0; }

# 2) 정말 모두 종료할지 묻기
read -r -p "Kill ALL above? [y/N] " ans
[[ $ans =~ ^[Yy]$ ]] || exit 0

# 3) 우선 SIGTERM, 남으면 SIGKILL
pgrep -f -- "$KW" | while read -r pid; do
  echo "TERM  $pid  $(ps -o cmd= -p "$pid")"
  kill -TERM "$pid" 2>/dev/null
done

sleep 1

for pid in $(pgrep -f -- "$KW"); do
  echo "KILL  $pid  $(ps -o cmd= -p "$pid")"
  kill -KILL "$pid" 2>/dev/null
done

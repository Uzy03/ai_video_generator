#!/bin/bash
# MLLB APIサーバをバックグラウンドで起動

LOGFILE="$(dirname "$0")/mlbb_api.log"

nohup uvicorn API.MLLB:app --host 0.0.0.0 --port 8000 > "$LOGFILE" 2>&1 &
echo "MLLB APIサーバをバックグラウンドで起動しました。ログ: $LOGFILE" 
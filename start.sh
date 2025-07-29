# 引数チェック
if [ "$#" -ne 2 ]; then
  echo "Usage: ./start.sh <theme> <turn>"
  exit 1
fi

# 引数を変数に代入
THEME="$1"
TURN="$2"
DIR="./outputs"

# 日時取得（例：2025-07-24-04-24）
TODAY=$(date +"%Y-%m-%d_%H-%M")

# 出力ファイル名
FT_OUTPUT_PATH="${DIR}/${TODAY}FT.json"
PT_OUTPUT_PATH="${DIR}/${TODAY}PT.json"
# 出力ディレクトリが存在しなければ作成
mkdir -p "${DIR}"

# APIサーバーにPOSTリクエストし、結果をファイルに出力
curl -X POST "http://127.0.0.1:8000/discussion/" \
  -H "Content-Type: application/json" \
  -d "{\"theme\": \"$THEME\", \"turn\": $TURN}" > "${FT_OUTPUT_PATH}"

# APIサーバーにPOSTリクエストし、結果をファイルに出力
curl -X POST "http://127.0.0.1:8001/discussion/" \
  -H "Content-Type: application/json" \
  -d "{\"theme\": \"$THEME\", \"turn\": $TURN}" > "${PT_OUTPUT_PATH}"

echo "✅ 出力完了: outputs"
# 引数チェック
if ($args.Count -ne 2) {
    Write-Host "Usage: .\start.ps1 <theme> <turn>"
    exit 1
}

# 引数を変数に代入
$THEME = $args[0]
$TURN = $args[1]
$DIR = "../outputs"

# 日時取得（例：2025-07-24_04-24）
$TODAY = Get-Date -Format "yyyy-MM-dd_HH-mm"

# 出力ファイルパス
$FT_OUTPUT_PATH = "$DIR/${TODAY}FT.json"
$PT_OUTPUT_PATH = "$DIR/${TODAY}PT.json"

# 出力ディレクトリを作成（存在しなければ）
if (-Not (Test-Path -Path $DIR)) {
    New-Item -ItemType Directory -Path $DIR | Out-Null
}

# APIサーバーにPOSTリクエストし、結果をファイルに出力
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/discussion/" `
    -ContentType "application/json" `
    -Body (@{ theme = $THEME; turn = [int]$TURN } | ConvertTo-Json -Depth 3) `
    | ConvertTo-Json -Depth 10 > $FT_OUTPUT_PATH

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8001/discussion/" `
    -ContentType "application/json" `
    -Body (@{ theme = $THEME; turn = [int]$TURN } | ConvertTo-Json -Depth 3) `
    | ConvertTo-Json -Depth 10 > $PT_OUTPUT_PATH

Write-Host "✅ 出力完了: outputs"
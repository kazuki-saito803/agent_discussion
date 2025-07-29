# agent_discussion

## 概要  
このアプリは、**微調整されたAIエージェントとプロンプトベースのエージェントの会話を可視化**する比較アプリです。  
また、モデルやその他の条件を変更することで、**エージェントの挙動の変化**を視覚的に確認することもできます。

## 目的  
調整の有無が**応答のスタイル、口調、出力の質にどのような影響を与えるか**を分析することを目的としています。

## 使用技術  

| 言語／ライブラリ／ツール | 使用用途 |
|:--------------------------|:----------|
| Python                    | アプリの基本ロジックの構築に使用 |
| PyTorch                   | 学習時に勾配を無効化するために使用 |
| FastAPI                   | 作成したモデルをAPIとして公開するために使用 |
| transformers              | モデルやトークナイザーの読み込みに使用 |
| LoRA FT（peft）           | LoRAで学習したアダプタをベースモデルに組み込むために使用 |
| Docker                    | APIの実行基盤となるコンテナを提供 |
| Hugging Face              | LoRAを用いたFTモデルなどを取得するために使用 |

## プロジェクト構成
```
agent_discussion/
├── README.md
├── requirements.txt
├── .gitignore
├── docker-compose.yml
├── start.sh
├── start.ps1
├── agents/
│   ├── finetuned/
│   │   ├── <agent>/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── models.py
│   └── prompt/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── templates.py
│   │   └── models.py
├── learning_material
│   └── training.ipynb/
├── datasets/
│   ├── <agent>.json
└── outputs/
```
## 使い方
1. Gitのインストール
1. プロジェクトのpull
1. Docker Desktopのインストール
1. Hugging Faceのログイン  
    以下のリンクからHugging Faceにログイン。アカウントがない場合はアカウント作成
    [https://huggingface.co/](https://huggingface.co/)
1. 使用モデルのAccess requestを行う
1. アクセストークン作成
1. docker-composeによるDockerイメージのビルドとコンテナ立ち上げ  
    以下のコマンドを実行する。
    ```
    docker-compose up
    ```
1. スクリプト実行
1. カスタム

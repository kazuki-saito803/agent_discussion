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

## 使い方
1. プロジェクトのpull
    このプロジェクトを格納するディレクトリまで移動したら以下のコマンドを実行する。
    ```bash
    git clone https://github.com/kazuki-saito803/agent_discussion.git
    ```
    または以下を実行する。
    ```bash
    git clone git@github.com:kazuki-saito803/agent_discussion.git
    ```
1. Docker Desktopのインストール
    Docker Desktopをインストールしてデーモンのプロセスを立ち上げる(Docker Desktopを立ち上げればプロセスは立ち上がります)
1. Hugging Faceのログイン  
    以下のリンクからHugging Faceにログイン。アカウントがない場合はアカウント作成
    [https://huggingface.co/](https://huggingface.co/)
1. アクセストークン作成
1. 使用モデルのAccess requestを行う
    今回、meta-llama/Llama-3.2-1B-Instructを使用するため以下のサイトから
1. Python仮想環境の作成
    ```bash
    python -m venv venv
    ```
1. Pytho外部ライブラリのインストール
    ```bash
    pip install -r requirements.txt
    ```
1. docker-composeによるDockerイメージのビルドとコンテナ立ち上げ  
    以下のコマンドを実行する。
    ```
    docker-compose up
    ```
1. スクリプト実行
    - Terminalで実行する場合
    ```bash
    start.sh "<生成AIに投げたいプロンプト>" 会話の回数
    ```
    - PowerShellで実行する場合
    ```bash
    start.ps1 "<生成AIに投げたいプロンプト>" 会話の回数
    ```
1. 実行結果の確認
    outputsディレクトリ配下に作成されたファイルから実行結果を確認する。
    - 実行日時PT.jsonと記載があるファイルがプロンプトチューニングした結果を格納したファイル
    - 実行日時FT.jsonと記載があるファイルがファインチューニングで実行した結果を格納したファイル
1. カスタム(例)
    - モデルカスタム
    agent_discussion/agents/finetuned/models.pyとagent_discussion/agents/prompt/models.pyのmodel_nameを対応したものに変えて別途、LoRAを行いそれぞれのディレクトリ配下に格納する。
    - 温度をはじめとした各種パラメータ
    finetunedのmodels.pyでself.model.generate内の引数を変更。promptのmodels.pyでpipeline内の引数を変更
    - 発言の順番を変更
    agent、promptそれぞれのDiscussion関数内にあるagent_instanceの順番を変更
    - エージェントに取り込ませるプロンプトを変更
    prompt/models.pyやagent/main.pyのpromptを変更
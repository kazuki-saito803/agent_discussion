from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# 🔸 LoRAアダプタの保存先（あなたのフォルダパス）
lora_adapter_path = "./educator/"

# 🔹 LoRA設定ファイルからベースモデル名を取得
peft_config = PeftConfig.from_pretrained(lora_adapter_path)
base_model_name = peft_config.base_model_name_or_path  # 例: "meta-llama/Llama-2-7b-hf"

# 🔸 ベースモデル読み込み（FP16 / GPU最適化）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",           # GPU 自動割当
    torch_dtype=torch.float16
)

# 🔹 LoRAアダプタをマージ（統合モデルを作成）
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model = model.merge_and_unload()  # LoRAとベースモデルを統合

# 🔸 トークナイザーはベースモデルと同じ
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 🔹 推論（例）
instruction = "プロジェクトに関する質問に答えてください"
input_text = "このプロジェクトでコンポーネントを格納するディレクトリ構成は？"
prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 🔸 出力表示
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "### Response:" in full_response:
    response = full_response.split("### Response:")[-1].strip()
else:
    response = full_response.strip()

print(response)

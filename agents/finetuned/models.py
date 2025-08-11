import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


load_dotenv(dotenv_path="../../.env")

token = os.getenv("HUGGINGFACE_TOKEN")

model_name = "meta-llama/Llama-3.2-1B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # GPU 自動割当
    torch_dtype=torch.float16,
    use_auth_token=token
)

class agent():
    def __init__(self, folder_name):
        self.role = folder_name
        self.folder_path = f"./{folder_name}"
        self.instruction = {
                "educator":"このテーマについて、生徒の学びやすさや理解の深まりという観点から、どのようにアプローチすべきか考えてください。教師としての経験を踏まえ、現場での実践可能性についても述べてください。",
                "engineer":"この課題を技術的に解決するにはどのような方法があるか、AIやITの活用可能性を中心に、効率性・実現性・革新性の観点から提案してください。データやロジックに基づく説明をお願いします。",
                "ethics_committee":"このアイデアや技術の導入に関して、倫理的なリスクや社会的な影響をどう評価しますか？プライバシー、人権、偏見の可能性などを考慮した上で、問題点とその対処策を挙げてください。",
                "gurdian":"この取り組みが子どもたちにどのような影響を与えるか、保護者の視点から考えてください。安全性、成長への影響、家庭でのサポートのしやすさについても意見を述べてください。",
                "student":"この話題について、学生の立場からリアルな感覚で意見を述べてください。実際に使うとしたらどう感じるか、何がうれしくて、何が不安か、身近な経験も交えて話してください。"
            }.get(folder_name)

        # 🔹 LoRAアダプタをマージ（統合モデルを作成）
        self.model = PeftModel.from_pretrained(base_model, self.folder_path)
        self.model = self.model.to(torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")  # 明示的にロード

        # 🔸 トークナイザーはベースモデルと同じ
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_auth_token=token)
    
        
    def predict(self, prompt):
        prompt = f"### Instruction:\n{self.instruction}\n\n### Input:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 🔸 出力表示
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response.strip()

        return response
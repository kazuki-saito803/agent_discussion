from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


model_name = "meta-llama/Llama-3.2-1B-Instruct"

# トークナイザーとモデルを読み込む
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},           # GPU使用（なければCPU）
    torch_dtype="auto"           # 自動で float16 などに切り替え（効率化）
)

# pipelineを構築
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,          # 生成する最大トークン数
    temperature=0.7,             # 出力の多様性（小さいとより決定的）
    top_p=0.9,                   # nucleus sampling の確率
    repetition_penalty=1.1       # 繰り返し抑制
)

def predict(theme, template, is_first, last_comment=""):
    if is_first:
        prompt = f"{template}\n\n{theme}についてあなたの意見を聞かせてください。"
    else:
        prompt = f"ある方は「{last_comment}」と言っています。{theme}についてあなたの意見を聞かせてください。"
    output = llama_pipeline(prompt)
    return output[0]["generated_text"]
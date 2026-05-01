Instruction (bash):
``` bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python api_server.py
```

Ссылка на модель Qwen/Qwen2-0.5B-Instruct - https://huggingface.co/Qwen/Qwen2-0.5B-Instruct.

Код для загрузки модели (python):
``` python
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import gc

# Путь к датасету из лабораторной работы №1
dataset_path = "corpus_variant_21"

# Загрузка датасета
dataset = load_from_disk(dataset_path)
print(f"Загружено примеров: {len(dataset)}")
print(f"Колонки: {dataset.column_names}")

model_name = "Qwen/Qwen2-0.5B-Instruct"  # или другая модель

# Конфигурация 4-битной квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Конфигурация LoRA
lora_config = LoraConfig(
    r=16,                      # ранг LoRA (обычно 8, 16, 32)
    lora_alpha=32,             # масштабирующий коэффициент
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # целевые модули
    lora_dropout=0.1,          # dropout для регуляризации
    bias="none",               # не обучать bias
    task_type=TaskType.CAUSAL_LM
)

# Подготовка модели к k-bit обучению
model = prepare_model_for_kbit_training(model)

# Применение LoRA к модели
model = get_peft_model(model, lora_config)

# Вывод информации о trainable параметрах
model.print_trainable_parameters()

def format_example(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read and understand the text: {example['text']}"}
    ]

    # Автоматически применяем правильный шаблон для Qwen
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted}

# Применение форматирования
formatted_dataset = dataset.map(format_example)
print(formatted_dataset[0]["text"][:200])

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./lora_adapters",         # директория для сохранения
    num_train_epochs=15,                  # количество эпох
    per_device_train_batch_size=2,        # размер батча (уменьшить при OOM)
    gradient_accumulation_steps=4,        # накопление градиентов
    warmup_steps=100,                     # шаги разогрева
    learning_rate=5e-4,                   # скорость обучения
    logging_steps=1,                      # частота логирования
    save_steps=100,                       # частота сохранения
    report_to="none"                      # отключить wandb
)

# Создание SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args
)

print("="*60)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*60)
print(f"Количество эпох: {training_args.num_train_epochs}")
print(f"Размер батча: {training_args.per_device_train_batch_size}")
print(f"Накопление градиентов: {training_args.gradient_accumulation_steps}")
print(f"Эффективный батч: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print("="*60)

# Запуск обучения
trainer.train()

import json

# Извлекаем логи
log_history = trainer.state.log_history

# Сохраняем логи в файл
with open("training_loss.json", "w") as f:
    json.dump(log_history, f, indent=4)

print("\n Обучение завершено!")

# Сохранение только LoRA-весов
lora_path = "./lora_adapter_fully_connected"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)

print(f" LoRA-адаптер сохранен в {lora_path}")

# Размер сохраненных файлов
total_size = 0
for file in os.listdir(lora_path):
    file_path = os.path.join(lora_path, file)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path)
        total_size += size
        print(f"  {file}: {size:,} bytes")
print(f"  ИТОГО: {total_size:,} bytes ({total_size/1024:.1f} KB)")

from peft import PeftModel

# Очистка памяти
gc.collect()
torch.cuda.empty_cache()

# Загрузка базовой модели (можно в 4-bit)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Загрузка LoRA-адаптера
fine_tuned_model = PeftModel.from_pretrained(base_model, lora_path)

# Тестирование
test_prompts = [
    f"The main idea of attention mechanism in LLM is",
    f"How does the attention mechanism improve upon the limitations of the traditional RNN-based encoder-decoder architecture?",
    f"What is the masked language model (MLM) and the next sentence prediction (NSP)?"
]

print("\n" + "="*60)
print("ТЕСТИРОВАНИЕ ДООБУЧЕННОЙ МОДЕЛИ")
print("="*60)

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = fine_tuned_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nПромпт: {prompt}")
    print(f"Ответ: {response}")
    print("-"*50)

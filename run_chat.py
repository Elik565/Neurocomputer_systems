from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./"  # Текущая директория с файлами модели
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Настройка технического токена
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("Enter: ")
    if user_input.lower() in ["exit", "quit", "стоп", "выход"]:
        break

    # Добавляем сообщение пользователя в список
    messages.append({"role": "user", "content": user_input})

    # Генерируем промпт в формате Qwen
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True # ВАЖНО: это ставит тег <|im_start|>assistant
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Генерируем ответ
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2
    )

    # Декодируем только ответ (отрезаем промпт)
    # Берем токены, начиная с длины входящего промпта
    answer_tokens = outputs[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    print("AI:", answer)

    # Добавляем ответ AI в историю (чтобы модель помнила контекст)
    messages.append({"role": "assistant", "content": answer})
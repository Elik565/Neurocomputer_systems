from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------ 1. Инициализация FastAPI ------------------
app = FastAPI(title="DialoGPT Chat API", version="1.0")

# Монтируем папку static для раздачи HTML, CSS, JS файлов
# Все файлы из папки ./static будут доступны по URL /static/...
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------ 2. Глобальная загрузка модели ------------------
# Модель загружается ОДИН РАЗ при старте сервера, а не при каждом запросе.
# Это критически важно для производительности (экономит десятки секунд на запрос).

model_path = "./"  # Путь к папке с вашей дообученной моделью
print("[INFO] Загрузка токенизатора и модели...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()  # Переводим в режим инференса
print("[INFO] Модель успешно загружена и готова к запросам.")

# ------------------ 3. Pydantic модели (валидация JSON) ------------------
class ChatMessage(BaseModel):
    """Структура одного сообщения в истории диалога"""
    user: str
    bot: str

class ChatRequest(BaseModel):
    """Структура входящего POST-запроса на /chat"""
    message: str                     # Текущий вопрос пользователя
    history: list[ChatMessage] = []  # История предыдущих реплик

class ChatResponse(BaseModel):
    """Структура ответа сервера"""
    response: str

# ------------------ 4. Обработчик главной страницы ------------------
@app.get("/")
async def read_root():
    """Возвращает index.html при заходе в корень сайта"""
    return FileResponse("static/index.html")

# ------------------ 5. Основной API-эндпоинт ------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Используем историю для создания списка сообщений
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in request.history:
        messages.append({"role": "user", "content": msg.user})
        messages.append({"role": "assistant", "content": msg.bot})
    
    # Добавляем текущий вопрос
    messages.append({"role": "user", "content": request.message})

    # Правильное форматирование для Qwen
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Используем run_in_threadpool, чтобы не "вешать" API
    def generate_sync():
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7
            )
        # Возвращаем только сгенерированную часть
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    response_text = await run_in_threadpool(generate_sync)
    return ChatResponse(response=response_text)

# ------------------ 6. Точка входа ------------------
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" позволяет подключаться к серверу с других устройств в локальной сети
    uvicorn.run(app, host="0.0.0.0", port=8000)
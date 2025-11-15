# inference_example.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "rut5_finetuned_epoch3.pt"

print(f"Загружаем модель из: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем: {device}")

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-base", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("cointegrated/rut5-base").to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Модель загружена!")
except FileNotFoundError:
    print("Файл не найден! Сначала обучите модель.")
    exit(1)

model.eval()


def generate_t5(text: str, max_len: int = 128) -> str:
    prefixed_text = f"summarize: {text}"
    inputs = tokenizer(
        prefixed_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=10,
            num_beams=5,
            length_penalty=0.6,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("=" * 60)
    print("   ИИ-СУММАРИЗАТОР (ruT5 + GPU)")
    print("=" * 60)

    while True:
        txt = input("\nВведите текст (или 'выход'): ").strip()
        if txt.lower() in ["выход", "exit"]:
            break
        if not txt:
            continue
        print("→", generate_t5(txt))
# fit_model.py
import torch
import glob
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from config import MODEL_NAME, PROC_DIR, BATCH_SIZE, GRAD_ACCUM_STEPS, EPOCHS, LR, MODELS_DIR
from utils import set_seed

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


class SimpleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    src_texts = [ex["src"] for ex in batch]
    tgt_texts = [ex["tgt"] for ex in batch]

    prefixed_texts = [f"summarize: {text}" for text in src_texts]

    inputs = tokenizer(
        prefixed_texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    labels = tokenizer(
        tgt_texts,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )["input_ids"]

    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "labels": labels.to(device)
    }


def train_one_epoch(loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Train")):
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
    return total_loss / len(loader)


if __name__ == "__main__":
    files = glob.glob(str(PROC_DIR / "*/train.jsonl"))
    train_examples = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                train_examples.append(json.loads(line.strip()))
    print(f"Загружено {len(train_examples)} примеров для обучения")

    if len(train_examples) == 0:
        raise ValueError("Нет данных! Запустите data_prepare_main.py")

    train_ds = SimpleDataset(train_examples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(train_loader, optimizer, scheduler)
        print(f"Epoch {epoch} | loss {loss:.4f}")
        save_path = MODELS_DIR / f"rut5_finetuned_epoch{epoch}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Сохранено: {save_path}")

    print("Обучение завершено! Модель готова.")
# ru_t5_optimize.py
import optuna
from fit_model import train_one_epoch, collate_fn
from torch.utils.data import DataLoader
from config import BATCH_SIZE, EPOCHS, PROC_DIR
import torch, pandas as pd, glob

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    bs = trial.suggest_categorical("batch_size", [4, 8, 16])
    files = glob.glob(str(PROC_DIR / "*/train.jsonl"))
    samples = []
    for f in files:
        df = pd.read_json(f, lines=True).sample(500, random_state=42)
        samples.extend(df.to_dict(orient="records"))
    loader = DataLoader(samples, batch_size=bs, shuffle=True, collate_fn=collate_fn)

    model = AutoModelForSeq2SeqLM.from_pretrained("cointegrated/rut5-base").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    loss = train_one_epoch(loader, optimizer, scheduler)
    return loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print(study.best_params)
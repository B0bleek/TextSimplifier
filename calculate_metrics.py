# calculate_metrics.py
import pandas as pd
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import pipeline
from config import PROC_DIR

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def calc_metrics(preds, refs):
    rouge1, rouge2, rougeL = [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    return {
        "rouge1": sum(rouge1)/len(rouge1),
        "rouge2": sum(rouge2)/len(rouge2),
        "rougeL": sum(rougeL)/len(rougeL)
    }

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(".", local_files_only=True)  # папка с весами
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    test_files = list((PROC_DIR / "gazeta").glob("test.jsonl"))
    df = pd.read_json(test_files[0], lines=True)

    preds = []
    for txt in tqdm(df["src"], desc="Predict"):
        out = pipe(txt, max_length=128, min_length=5, do_sample=False)[0]["summary_text"]
        preds.append(out)

    metrics = calc_metrics(preds, df["tgt"].tolist())
    print(metrics)
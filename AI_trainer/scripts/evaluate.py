from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import Dataset
from load_data import load_and_prepare_datasets
from preprocess import tokenize_and_wrap
from metrics import compute_metrics

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent

    train_path = repo_root / "dataset" / "train.txt"
    val_path = repo_root / "dataset" / "val.txt"
    test_path = repo_root / "dataset" / "test.txt"

    _, _, df_test, label2id, id2label = load_and_prepare_datasets(
        str(train_path), str(val_path), str(test_path)
    )
    labels = [id2label[i] for i in range(len(label2id))]

    model_dir = repo_root / "model"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    test_dataset = tokenize_and_wrap(tokenizer, df_test)

    trainer = Trainer(model=model)

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)

    print("Evaluation Metrics on Test Set:")
    print(f"Accuracy: {(y_pred == y_true).mean():.4f}")
    print(f"F1-score (weighted): {compute_metrics((logits, y_true))['f1']:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()

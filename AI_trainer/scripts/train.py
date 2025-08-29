from transformers import TrainingArguments, EarlyStoppingCallback
from weighted_trainer import WeightedTrainer
from load_data import load_and_prepare_datasets
from preprocess import tokenize_and_wrap
from model_setup import load_model_and_tokenizer
from metrics import compute_metrics
from PlotMetrics import PlotMetricsCallback
from pathlib import Path
import torch
import numpy as np

def main():
    model_name = "distilbert-base-uncased"

    # Resolve project root as the parent of this file's directory
    repo_root = Path(__file__).resolve().parent.parent

    train_path = repo_root / "dataset" / "train.txt"
    val_path = repo_root / "dataset" / "val.txt"
    test_path = repo_root / "dataset" / "test.txt"

    df_train, df_val, df_test, label2id, id2label = load_and_prepare_datasets(
        str(train_path), str(val_path), str(test_path)
    )

    tokenizer, model = load_model_and_tokenizer(model_name, len(label2id), label2id, id2label)

    train_dataset = tokenize_and_wrap(tokenizer, df_train)
    val_dataset = tokenize_and_wrap(tokenizer, df_val)

    model_dir = repo_root / "model"

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_total_limit=2,
        num_train_epochs=5  ,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.05,
        learning_rate=2e-5,
        warmup_steps= 100,
        max_grad_norm=1.0,
        logging_dir=str(repo_root / "logs"),
        load_best_model_at_end = True,
        metric_for_best_model="eval_loss",
        greater_is_better= False,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
        save_safetensors=False,  # avoid Windows file lock issues
    )

    # Compute per-class pos_weight = N_neg/N_pos (handles imbalance)
    # df_train['label'] is multi-hot vectors
    labels_array = np.stack(df_train['label'].to_list())  # shape [N, C]
    pos_counts = labels_array.sum(axis=0)
    neg_counts = labels_array.shape[0] - pos_counts
    # Avoid division by zero
    pos_counts = np.where(pos_counts == 0, 1, pos_counts)
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[PlotMetricsCallback(str(model_dir)), EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=pos_weight,
    )

    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

if __name__ == "__main__":
    main()

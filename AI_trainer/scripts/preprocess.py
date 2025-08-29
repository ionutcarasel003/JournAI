from datasets import Dataset


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
    )


def tokenize_and_wrap(tokenizer, df):
    df = df[["text", "label"]].copy()
    dataset = Dataset.from_pandas(df)
    # Batched tokenization
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])  # drop raw text
    # Rename label column to 'labels' for HF Trainer
    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")
    # Set tensor format for Trainer
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

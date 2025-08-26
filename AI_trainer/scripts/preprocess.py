from datasets import Dataset

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length"
    )

def tokenize_and_wrap(tokenizer, df):
    df = df[['text', 'label']].copy()
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    return dataset

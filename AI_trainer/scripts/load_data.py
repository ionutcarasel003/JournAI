import pandas as pd

def load_and_prepare_datasets(train_path, val_path, test_path):
    # Load TSV-like text files with semicolon separator and two columns: text;label
    df_train = pd.read_csv(train_path, sep=';', names=['text', 'label'], header=None)
    df_val = pd.read_csv(val_path, sep=';', names=['text', 'label'], header=None)
    df_test = pd.read_csv(test_path, sep=';', names=['text', 'label'], header=None)

    # Label encoding
    labels = sorted(df_train['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Map
    for df in [df_train, df_val, df_test]:
        df['label'] = df['label'].map(label2id)

    return df_train, df_val, df_test, label2id, id2label

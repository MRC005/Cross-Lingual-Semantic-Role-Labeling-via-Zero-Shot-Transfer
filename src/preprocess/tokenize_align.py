from datasets import Dataset
from transformers import AutoTokenizer
from src.preprocess.up_parser import load_srl_dataset

MODEL_NAME = "bert-base-multilingual-cased"

label2id = {
    "O": 0,

    "B-ARG0": 1,
    "I-ARG0": 2,

    "B-ARG1": 3,
    "I-ARG1": 4,

    "B-ARG2": 5,
    "I-ARG2": 6,

    "B-ARG3": 7,
    "I-ARG3": 8,

    "B-ARG4": 9,
    "I-ARG4": 10,

    "B-ARG5": 11,
    "I-ARG5": 12,

    "B-ARGM": 13,
    "I-ARGM": 14,
}

id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def examples_to_dataset(examples):
    return Dataset.from_dict({
        "tokens": [e["words"] for e in examples],
        "labels": [e["labels"] for e in examples],
    })


def normalize_label(label):
    """
    Converts:
    B-ARG1-DSP -> B-ARG1
    B-ARGM-TMP -> B-ARGM
    """
    if label == "O":
        return label

    parts = label.split("-")

    if len(parts) >= 2:
        return parts[0] + "-" + parts[1]

    return label


def tokenize_and_align(examples):

    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    aligned_labels = []

    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word = None
        label_ids = []

        for word_id in word_ids:

            if word_id is None:
                label_ids.append(-100)

            elif word_id != prev_word:

                raw_label = label_seq[word_id]
                clean_label = normalize_label(raw_label)

                # 🔥 Safe fallback for rare roles (ARGA, etc.)
                if clean_label not in label2id:
                    if clean_label.startswith("B-"):
                        clean_label = "B-ARGM"
                    elif clean_label.startswith("I-"):
                        clean_label = "I-ARGM"
                    else:
                        clean_label = "O"

                label_ids.append(label2id[clean_label])

            else:
                label_ids.append(-100)

            prev_word = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


if __name__ == "__main__":

    train = load_srl_dataset(
        "UniversalPropositions/UP_English-EWT/en_ewt-up-train.conllu"
    )

    train_ds = examples_to_dataset(train)

    tokenized_train = train_ds.map(
        tokenize_and_align,
        batched=True
    )

    print(tokenized_train[0])
    print("\nLabels example:\n", tokenized_train[0]["labels"][:20])
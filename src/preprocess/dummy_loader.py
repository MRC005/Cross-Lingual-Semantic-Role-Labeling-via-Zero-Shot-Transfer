from transformers import AutoTokenizer
from src.preprocess.labels import label2id

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

DATA_PATH = "data/dummy_srl.txt"

def load_data(path):
    sentences = []
    labels = []

    with open(path, "r") as f:
        lines = f.read().strip().split("\n\n")

    for block in lines:
        sent, lab = block.split("\n")
        sentences.append(sent.split())
        labels.append(lab.split())

    return sentences, labels


def tokenize_and_align(sentences, labels):
    encodings = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    all_labels = []

    for i, label_seq in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        prev_word = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                label_ids.append(label2id[label_seq[word_id]])
            else:
                label_ids.append(-100)

            prev_word = word_id

        all_labels.append(label_ids)

    return encodings, all_labels


if __name__ == "__main__":
    sents, labs = load_data(DATA_PATH)
    enc, aligned = tokenize_and_align(sents, labs)

    print("Sentence:", sents[0])
    print("Tokens:", tokenizer.convert_ids_to_tokens(enc["input_ids"][0]))
    print("Label IDs:", aligned[0])
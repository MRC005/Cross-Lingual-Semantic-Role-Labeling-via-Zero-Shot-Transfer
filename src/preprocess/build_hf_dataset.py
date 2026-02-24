from datasets import Dataset
from src.preprocess.up_parser import load_srl_dataset


def examples_to_dataset(examples):
    return Dataset.from_dict({
        "tokens": [e["words"] for e in examples],
        "labels": [e["labels"] for e in examples],
    })


if __name__ == "__main__":

    train = load_srl_dataset(
        "UniversalPropositions/UP_English-EWT/en_ewt-up-train.conllu"
    )

    dev = load_srl_dataset(
        "UniversalPropositions/UP_English-EWT/en_ewt-up-dev.conllu"
    )

    test = load_srl_dataset(
        "UniversalPropositions/UP_English-EWT/en_ewt-up-test.conllu"
    )

    train_dataset = examples_to_dataset(train)
    dev_dataset = examples_to_dataset(dev)
    test_dataset = examples_to_dataset(test)

    print("Train size:", len(train_dataset))
    print("Dev size:", len(dev_dataset))
    print("Test size:", len(test_dataset))

    print("\nSample example:")
    print(train_dataset[0])
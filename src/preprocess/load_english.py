from datasets import load_dataset

def load_english_srl():
    # Load the UP2.0 universal propositions dataset
    dataset = load_dataset("universal_propositions_hierarchy", "en")

    print(dataset)

    sample = dataset["train"][0]
    print("\nSample keys:", sample.keys())
    print("\nWords:", sample["tokens"])
    print("\nSRL labels:", sample["labels"])

if __name__ == "__main__":
    load_english_srl()
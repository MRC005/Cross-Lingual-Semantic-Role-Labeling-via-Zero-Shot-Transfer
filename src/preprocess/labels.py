# SRL Label Schema (Simplified PropBank-style)

LABEL_LIST = [
    "O",
    "B-ARG0", "I-ARG0",
    "B-ARG1", "I-ARG1",
    "B-ARG2", "I-ARG2",
    "B-ARGM", "I-ARGM"
]

# Create mappings

label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for label, idx in label2id.items()}

NUM_LABELS = len(LABEL_LIST)

if __name__ == "__main__":
    print("Label to ID mapping:")
    print(label2id)
    print("\nTotal labels:", NUM_LABELS)
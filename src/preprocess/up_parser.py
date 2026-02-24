def parse_up_file(filepath):
    sentences = []
    current_tokens = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            if line == "":
                if current_tokens:
                    sentences.append(current_tokens)
                    current_tokens = []
                continue

            cols = line.split("\t")

            if "-" in cols[0] or "." in cols[0]:
                continue

            token = {
                "id": int(cols[0]),
                "form": cols[1],
                "pred_sense": cols[10] if len(cols) > 10 else "_",
                "arg_cols": cols[11:] if len(cols) > 11 else []
            }

            current_tokens.append(token)

    if current_tokens:
        sentences.append(current_tokens)

    return sentences


def convert_to_bio(raw_labels):

    def normalize(label):
        if label in ("_", "V", "C-V"):
            return "O"
        if label.startswith("C-") or label.startswith("R-"):
            label = label[2:]
        if label.startswith("ARGM"):
            return "ARGM"
        if label.startswith("ARG"):
            return label
        return "O"

    bio = []
    prev = "O"

    for lab in raw_labels:
        role = normalize(lab)

        if role == "O":
            bio.append("O")
            prev = "O"
        elif role != prev:
            bio.append(f"B-{role}")
            prev = role
        else:
            bio.append(f"I-{role}")

    return bio


def sentence_to_examples(tokens):

    if len(tokens) == 0:
        return []

    num_preds = len(tokens[0]["arg_cols"])
    if num_preds == 0:
        return []

    words = [t["form"] for t in tokens]
    examples = []

    for p in range(num_preds):
        raw = [t["arg_cols"][p] for t in tokens]
        labels = convert_to_bio(raw)

        examples.append({
            "words": words,
            "labels": labels
        })

    return examples


def load_srl_dataset(filepath):

    sentences = parse_up_file(filepath)
    all_examples = []

    for sent in sentences:
        ex = sentence_to_examples(sent)
        all_examples.extend(ex)

    return all_examples
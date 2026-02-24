from conllu import parse_incr


def load_up_file(filepath):
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            sentences.append(sent)
    return sentences


if __name__ == "__main__":
    path = "UniversalPropositions/UP_English-EWT/en_ewt-up-train.conllu"
    sentences = load_up_file(path)

    print("Total sentences:", len(sentences))

    found = False

    for sent in sentences:
        for tok in sent:
            misc = tok.get("misc")
            if misc and "srl" in misc:
                print("\nFOUND SRL SENTENCE\n")
                for t in sent:
                    print(dict(t))
                found = True
                break
        if found:
            break

    if not found:
        print("No SRL found in file")
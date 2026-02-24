from src.preprocess.up_parser import load_srl_dataset

data = load_srl_dataset(
    "UniversalPropositions/UP_English-EWT/en_ewt-up-train.conllu"
)

print("Total predicate examples:", len(data))

ex = data[0]
print("\nFirst example:\n")
for w, l in zip(ex["words"], ex["labels"]):
    print(w, l)
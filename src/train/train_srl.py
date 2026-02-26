import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import precision_score, recall_score, f1_score

from src.preprocess.up_parser import load_srl_dataset
from src.preprocess.tokenize_align import (
    tokenize_and_align,
    examples_to_dataset,
    label2id,
    id2label
)

def main():
    print("SCRIPT STARTED")
    print("Starting training...")

    MODEL_NAME = "bert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_examples = load_srl_dataset(
        "UniversalPropositions/UP_English-EWT/en_ewt-up-train.conllu"
    )
    dev_examples = load_srl_dataset(
        "UniversalPropositions/UP_English-EWT/en_ewt-up-dev.conllu"
    )

    train_ds = examples_to_dataset(train_examples)
    dev_ds = examples_to_dataset(dev_examples)

    train_tok = train_ds.map(tokenize_and_align, batched=True)
    dev_tok = dev_ds.map(tokenize_and_align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_labels = []
        true_preds = []
        for pred_seq, label_seq in zip(predictions, labels):
            sent_labels = []
            sent_preds = []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    sent_labels.append(id2label[l])
                    sent_preds.append(id2label[p])
            true_labels.append(sent_labels)
            true_preds.append(sent_preds)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    args = TrainingArguments(
        output_dir="./srl-model",
        eval_strategy="epoch",       
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        fp16=False,                   
        dataloader_num_workers=0,
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    trainer.save_model("./srl-model-final")
    tokenizer.save_pretrained("./srl-model-final")
    print("ALL DONE!")

if __name__ == "__main__":   
    main()
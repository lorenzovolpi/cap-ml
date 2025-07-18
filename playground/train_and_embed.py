import json
import os
import pdb
from dataclasses import dataclass
from itertools import batched

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import quacc as qc

base_dir = os.path.join(qc.env["OUT_DIR"], "transformers")


def get_tr_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join(base_dir, "models", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_embed_outdir(args):
    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    outdir = os.path.join(base_dir, "embeds", dataset_name, model_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def compute_clf_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    recall = recall_score(y_true=_labels, y_pred=_preds)
    precision = precision_score(y_true=_labels, y_pred=_preds)
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


def get_classifier(model_name, n_classes=2, use_bfloat16=False, device="cuda"):
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, torch_dtype=torch_dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_training_args(args):
    training_outdir = get_tr_outdir(args)
    trainer_args = TrainingArguments(
        output_dir=training_outdir,
        do_train=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.train_batchsize,
        per_device_eval_batch_size=args.train_batchsize * 4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        bf16=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        eval_on_start=True,
        load_best_model_at_end=True,
    )

    return trainer_args


def get_cls_bertlike(x):
    cls_emebds = x[
        :, 0, :
    ]  # get representation associated with the "CLS" token. In BERT-like models this is usually assigned with the first token of the input sequence (idx=0)
    return cls_emebds


def embed(model, tokenizer, data, selection_strategy, args):
    split_logits = []
    split_hidden_states = []

    for batch in batched(tqdm(data), n=args.embed_batchsize):
        texts, labels = zip(*((d["text"], d["label"]) for d in batch))
        with torch.no_grad():
            model_inputs = tokenizer(
                texts, truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
            )  # pad each batch to max_length
            output = model(**model_inputs.to(args.device), output_hidden_states=True)

        logits = output.logits
        hidden_states = output.hidden_states
        last_hidden_states = hidden_states[-1]

        split_hidden_states.append(selection_strategy(last_hidden_states.cpu().detach()))
        split_logits.append(logits.cpu().detach())

    split_logits = torch.vstack(split_logits)
    split_hidden_states = torch.vstack(split_hidden_states)

    return split_logits, split_hidden_states


def get_prevalence(labels):
    labels = torch.tensor(labels)
    unique_lbls = labels.unique()
    return [((labels == lbl).sum() / len(labels)).item() for lbl in unique_lbls]


def main(args):
    print(f"- model: {args.model_name}")
    print(f"- dataset: {args.dataset_name}")

    dataset = load_dataset(args.dataset_name)

    # create validation split if does not exist
    if "validation" not in dataset:
        print("splitting training set into train/validation...")
        _tmp_dataset = dataset["train"].train_test_split(test_size=args.val_size)
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]

    n_inferred_classes = dataset["train"].shape[-1]
    if args.num_classes != n_inferred_classes:
        print(
            f"number of inferred target classes ({n_inferred_classes}) != number of given target classes ({args.num_classes})"
        )

    model, tokenizer = get_classifier(model_name=args.model_name, n_classes=args.num_classes, device=args.device)

    # freeze base model -> train only fresh init classification head
    if not args.train_backbone:
        print("- freezing base model weights")
        for _, layer_weights in model.base_model.named_parameters():
            layer_weights.requires_grad = False

        # check trainable layers
        trainable_layers = []
        for layer_name, layer_weights in model.named_parameters():
            if layer_weights.requires_grad:
                trainable_layers.append(layer_name)
        print(f"- trainable layers: {trainable_layers}")

    def tokenize_dataset(sample):
        return tokenizer(
            sample["text"], truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
        )

    # tokenize dataset
    dataset = dataset.map(tokenize_dataset, batched=True, num_proc=8)

    # train model
    trainer_args = get_training_args(args)
    print(f"- storing model in {trainer_args.output_dir}")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=trainer_args,
        compute_metrics=compute_clf_metrics,
        callbacks=[],  # early stopping callback goes here, if needed
    )
    print("\nTraining...")
    trainer.train()

    print(torch.cuda.memory_allocated())
    # Get embedddings and logits
    embeds_outdir = get_embed_outdir(args)
    print("\nEmbedding...")
    print(f"- storing embeddings in {embeds_outdir}")
    splits = ["validation", "test"]
    for split in splits:
        split_data = dataset[split]
        split_logits, split_last_hiddens = embed(
            model, tokenizer, data=split_data, selection_strategy=get_cls_bertlike, args=args
        )
        split_labels = torch.tensor(dataset[split]["label"])

        torch.save(split_logits, os.path.join(embeds_outdir, f"logits.{split}.pt"))
        torch.save(split_last_hiddens, os.path.join(embeds_outdir, f"hidden_states.{split}.pt"))
        torch.save(split_labels, os.path.join(embeds_outdir, f"labels.{split}.pt"))

    with open(os.path.join(embeds_outdir, "dataset_info.json"), "w") as f:
        info = {"L_prev": get_prevalence(dataset["train"]["label"])}
        json.dump(info, f)


@dataclass
class MainArgs:
    model_name: str = "google-bert/bert-base-uncased"
    dataset_name: str = "stanfordnlp/imdb"
    num_classes: int = 2
    max_length: int = 512
    nepochs: int = 1
    train_batchsize: int = 64
    embed_batchsize: int = 512
    val_size: float = 0.2
    lr: float = 5e-4
    train_backbone: bool = True
    device: str = "cuda"


if __name__ == "__main__":
    args = MainArgs()
    main(args)

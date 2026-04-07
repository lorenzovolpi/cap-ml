import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from traceback import print_exception
from typing import Any, Callable, Iterator

import numpy as np
import quapy as qp
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCrop,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from cap_exp.data import ClassifierInfo, DatasetInfo, PretainInfo
from cap_exp.pretrain.dataset import get_hf_dataset, get_local_hf_dataset, save_dataset

EXPERIMENT = "pretrain"
DOMAIN = "image"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = True
qp.environ["_R_SEED"] = 0

# log = get_logger(id=f"{PROJECT}.{EXPERIMENT}.{DOMAIN}")

hf_dataset_map = {}
hf_model_map = {}


# class LoggingCallback(TrainerCallback):
#     filter_fields = set(
#         ["loss", "learning_rate", "eval_loss", "eval_acc", "eval_runtime", "train_runtime", "train_loss"]
#     )
#
#     def __init__(self, p: PretainInfo) -> None:
#         self.p = p
#
#     def get_logs_str(self, logs: dict):
#         strs = []
#         for k, v in logs.items():
#             if k == "learning_rate":
#                 strs.append(f"'{k}': {v:.4E}")
#             elif isinstance(v, int):
#                 strs.append(f"'{k}': {v}")
#             else:
#                 strs.append(f"'{k}': {v:.4f}")
#
#         return strs
#         # return [f"'{k}': {v:.4f}" if not isinstance(v, int) else f"'{k}': {v}" for k, v in logs.items()]
#
#     def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
#         if logs is None:
#             return
#
#         logs_with_step = {
#             "step": state.global_step,
#             "epoch": state.epoch,
#             **{k: v for k, v in logs.items() if k in self.filter_fields},
#         }
#         logs_str = "{" + ", ".join(self.get_logs_str(logs_with_step)) + "}"
#         log.info(f"[{self.p.h_info.name}@{self.p.d_info.name}] training log: {logs_str}")
#


def _fdataset(name: str, n: int) -> DatasetInfo:
    proper_name = name.replace("/", "__")
    hf_dataset_map[proper_name] = name
    return DatasetInfo(proper_name, DOMAIN, n)


def _fmodel(name: str, default: bool = True, **kwargs) -> ClassifierInfo:
    proprer_name = name.replace("/", "__")
    hf_model_map[proprer_name] = name
    return proprer_name


def sout(*args):
    if VERBOSE:
        print(*args)


def gen_datasets() -> Iterator[DatasetInfo]:
    yield _fdataset("mnist", 10)
    yield _fdataset("cifar10", 10)
    yield _fdataset("cifar100", 100)
    yield _fdataset("ethz/food101", 101)
    # yield _fdataset("caltech256", 257)
    yield _fdataset("imagenet-lt200", 200)
    yield _fdataset("imagenet-lt100", 200)
    yield _fdataset("cifar10-lt", 10)
    yield _fdataset("cifar100-lt", 100)
    yield _fdataset("food101-lt", 101)


# fmt: off
def gen_model_args(d_info: DatasetInfo) -> Iterator[ClassifierInfo]:
    def ovverride_params(model_name: str, args: VisionArgs, d_info: DatasetInfo):
        _overrides = {
            ("*", "mnist"): dict(nepochs=5, lr=5e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "cifar10"): dict(nepochs=5, lr=5e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "cifar100"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            # ("*", "caltech256"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "ethz/food101"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "imagenet-lt200"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "imagenet-lt100"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "cifar10-lt"): dict(nepochs=5, lr=5e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "cifar100-lt"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
            ("*", "food101-lt"): dict(nepochs=5, lr=3e-4, warmup_steps=200, train_bsize=64, train_hl=True, weight_decay=0.05),
        }

        d_name = hf_dataset_map.get(d_info.name, d_info.name)
        or_params = _overrides.get(("*", d_name), {}) | _overrides.get((model_name, d_name), {})
        return args.update(or_params)

    def mp(name: str, default=True, args=None):
        args = args if args else VisionArgs()
        return dict(name=name, default=default, args=args)

    model_params = [
        mp("microsoft/resnet-50"),
        mp("facebook/convnext-tiny-224"),
        mp("google/efficientnet-b0"),
        mp("google/vit-base-patch16-224"),
        mp("microsoft/swin-tiny-patch4-window7-224"),
    ]
    for mp in model_params:
        proper_name = _fmodel(mp["name"])
        args = ovverride_params(mp["name"], mp["args"], d_info)
        yield ClassifierInfo(class_name=proper_name, params=args.params, default=mp["default"])

# fmt: on


def gen_config():
    for d_info in gen_datasets():
        for h_info in gen_model_args(d_info):
            yield d_info, h_info


def get_tr_outdir(p_info: PretainInfo):
    outdir = os.path.join("output", "tms", "models", p_info.h_info.full_name, p_info.d_info.name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_val_split(dataset):
    _default = 0.5
    _val_splits = {
        "mnist": 0.65,
        "cifar10": 0.6,
        "cifar100": 0.6,
    }
    return _val_splits.get(hf_dataset_map.get(dataset, dataset), _default)


@dataclass()
class VisionArgs:
    max_length: int = 512
    nepochs: int = 2
    train_bsize: int = 32
    embed_bsize: int = 512
    lr: float = 2e-5
    train_hl: bool = False
    load_bf16: bool = False
    warmup_steps: int = 200
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    @property
    def params(self) -> dict[str, Any]:
        # NOTE: asdict makes a deepcopy!
        return asdict(self)

    def update(self, params: dict):
        return VisionArgs(**(self.params | params))


def get_dataset(d_info: DatasetInfo):
    d_name = hf_dataset_map.get(d_info.name, d_info.name)

    if d_name in ["mnist", "cifar10", "cifar100", "ethz/food101"]:
        dataset = get_hf_dataset(d_name)
    elif d_name in ["caltech256", "imagenet-lt200", "imagenet-lt100", "cifar10-lt", "cifar100-lt", "food101-lt"]:
        dataset = get_local_hf_dataset(d_name)

    return dataset


def create_transforms(image_processor, d_name, is_train=True):
    """
    Crea le trasformazioni per il dataset.
    """
    if isinstance(image_processor.size, dict):
        if "height" in image_processor.size:
            size = image_processor.size["height"]
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            # Prendi il primo valore disponibile
            size = list(image_processor.size.values())[0]
    else:
        # Se size è un intero diretto
        size = image_processor.size

    # Normalizzazione dal processor del modello
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    if is_train:
        if d_name == "mnist":
            transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
        elif d_name in ["cifar10", "cifar10-lt"]:
            transforms = Compose(
                [Resize(size), RandomCrop(size, padding=4), RandomHorizontalFlip(p=0.5), ToTensor(), normalize]
            )
        elif d_name in ["cifar100", "cifar100-lt"]:
            transforms = Compose(
                [
                    Resize(size),
                    RandomCrop(size, padding=4),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    ToTensor(),
                    normalize,
                    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                ]
            )
        elif d_name in ["ethz/food101", "food101-lt"]:
            transforms = Compose(
                [
                    RandomResizedCrop(size),
                    RandomHorizontalFlip(),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    ToTensor(),
                    normalize,
                ]
            )
        elif d_name == "ljnlonoljpiljm/caltech256":
            transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
        elif d_name == "imagenet-lt200":
            transforms = Compose(
                [
                    RandomResizedCrop(size),
                    RandomHorizontalFlip(),
                    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    ToTensor(),
                    normalize,
                ]
            )
        else:
            transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    else:
        transforms = Compose([Resize((size, size)), ToTensor(), normalize])

    return transforms


def preprocess_images(examples, transforms):
    """Preprocessa le immagini di training."""
    if "image" not in examples and "pixel_values" in examples:
        return examples

    # Gestisce sia MNIST (grayscale) che CIFAR (RGB)
    images = examples["image"]

    # Converti grayscale a RGB se necessario (per MNIST)
    processed_images = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        processed_images.append(transforms(img))

    examples["pixel_values"] = processed_images
    examples = {k: v for k, v in examples.items() if k not in ["image"]}
    return examples


def preprocess_dataset(args: VisionArgs, image_processor, dataset, d_name):
    d_name = hf_dataset_map.get(d_name, d_name)
    train_transforms = create_transforms(image_processor, d_name, is_train=True)
    val_transforms = create_transforms(image_processor, d_name, is_train=False)

    dataset["train"] = dataset["train"].with_transform(lambda examples: preprocess_images(examples, train_transforms))
    for split in ["validation", "test"]:
        dataset[split] = dataset[split].with_transform(lambda examples: preprocess_images(examples, val_transforms))

    return dataset


def get_classifier(model_name: str, d_info: DatasetInfo, args: VisionArgs):
    torch_dtype = torch.bfloat16 if args.load_bf16 else torch.float32
    model_name = hf_model_map.get(model_name, model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=d_info.n_classes,
        ignore_mismatched_sizes=True,
        torch_dtype=torch_dtype,
    ).to(DEVICE)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    return model, image_processor


def prepare_model(args: VisionArgs, model):
    if not args.train_hl:
        sout("- freezing base model weights")
        for _, layer_weights in model.base_model.named_parameters():
            layer_weights.requires_grad = False

        trainable_layers = []
        for layer_name, layer_weights in model.named_parameters():
            if layer_weights.requires_grad:
                trainable_layers.append(layer_name)
        sout(f"- trainable layers: {trainable_layers}")

    return model


def compute_clf_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "f1": f1}


def train_model(args: VisionArgs, p_info: PretainInfo, model, dataset, parser_args):
    training_outdir = get_tr_outdir(p_info)

    training_args = TrainingArguments(
        output_dir=training_outdir,
        do_train=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=4 * args.train_bsize,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        logging_steps=50,
        bf16=True,
        metric_for_best_model="acc",
        greater_is_better=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",  # Disabilita wandb/tensorboard se non configurati
        eval_on_start=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # train model
    sout(f"- storing model in {training_args.output_dir}")
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        compute_metrics=compute_clf_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-4),
            # LoggingCallback(p_info),
        ],
    )

    if parser_args.retrain:
        trainer.train()
    else:
        last_ckpt = get_last_checkpoint(training_outdir)
        if last_ckpt is None:
            sout("\nTraining...")
        else:
            sout("\nLoading last checkpoint...")
        trainer.train(resume_from_checkpoint=last_ckpt)

    return trainer, training_args


def extract_embeddings(outputs):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        last_hidden_state = outputs.hidden_states[-1]

        if len(last_hidden_state.shape) == 3:  # Transformer
            embeddings = last_hidden_state[:, 0].cpu().detach()  # CLS token
        elif len(last_hidden_state.shape) == 4:  # CNN
            embeddings = last_hidden_state.mean(dim=[2, 3]).cpu().detach()
        else:
            embeddings = last_hidden_state.cpu().detach()
    else:
        # Fallback: usa il pooler output se disponibile
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output.cpu().detach()
        else:
            embeddings = None

    return embeddings


def embed(model, data, selection_strategy: Callable, args: VisionArgs):
    # text_tag = "text"
    split_logits = []
    split_posteriors = []
    split_hidden_states = []
    split_y = []
    dataloader = DataLoader(
        # data.remove_columns(["image"]),
        data,
        batch_size=args.embed_bsize,
        shuffle=False,
        collate_fn=DefaultDataCollator(),
    )
    # for batch in batched(tqdm(data), n=args.embed_bsize):
    for batch in tqdm(dataloader, desc="Embedding"):
        # texts, labels = zip(*((d[text_tag], d["label"]) for d in batch))
        # labels = [d["label"] for d in batch]
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].cpu()
        with torch.no_grad():
            # model_inputs = tokenizer(
            #     texts, truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt"
            # )  # pad each batch to max_length
            # output = model(**model_inputs.to(DEVICE), output_hidden_states=True)
            output = model(**inputs, output_hidden_states=True)
        logits = output.logits
        posteriors = torch.softmax(logits, dim=-1)

        split_y.append(torch.tensor(labels))
        split_hidden_states.append(selection_strategy(output))
        split_logits.append(logits.cpu().detach())
        split_posteriors.append(posteriors.cpu().detach())

    split_y = torch.cat(split_y, dim=0).numpy()
    split_logits = torch.vstack(split_logits).numpy()
    split_posteriors = torch.vstack(split_posteriors).numpy()
    split_hidden_states = torch.vstack(split_hidden_states).numpy()

    return split_y, split_posteriors, split_hidden_states, split_logits


def pretrain(d_info: DatasetInfo, h_info: ClassifierInfo, parser_args):
    p_info = PretainInfo(domain=DOMAIN, d_info=d_info, h_info=h_info)
    if p_info.exists and not parser_args.ignore_exist:
        # log.info(f"[{h_info.name}@{d_info.name}] already exists, skipping.")
        return

    args = VisionArgs(**h_info.params)
    model_name = h_info.class_name

    # log.info(f"[{h_info.name}@{d_info.name}] started pretrain")
    sout(f"- model: {h_info.name}")
    sout(f"- dataset: {d_info.name}")

    dataset = get_dataset(d_info)
    # log.info(f"[{h_info.name}@{d_info.name}] dataset loaded")

    model, image_processor = get_classifier(model_name, d_info, args)
    model = prepare_model(args, model)
    # log.info(f"[{h_info.name}@{d_info.name}] model loaded")

    dataset = preprocess_dataset(args, image_processor, dataset, d_info.name)
    # log.info(f"[{h_info.name}@{d_info.name}] dataset pre-processed")

    train_model(args, p_info, model, dataset, parser_args)
    # log.info(f"[{h_info.name}@{d_info.name}] model trained")

    if parser_args.dry_run:
        return

    # Get embedddings and logits
    sout("\nEmbedding...")
    splits = ["validation", "test"]
    logits = {}
    embedddings = {}
    labels = {}
    posteriors = {}
    for split in splits:
        split_data = dataset[split]
        split_y, split_posteriors, split_last_hiddens, split_logits = embed(
            model, data=split_data, selection_strategy=extract_embeddings, args=args
        )
        embedddings[split] = split_last_hiddens
        posteriors[split] = split_posteriors
        labels[split] = split_y
        logits[split] = split_logits

    label_tag = "label"
    train_labels = np.array(dataset["train"][label_tag])
    classes = np.unique(train_labels)
    train_prev = np.sum(classes.reshape(-1, 1) == train_labels, axis=-1) / train_labels.shape[0]

    save_dataset(DOMAIN, d_info.name, h_info.full_name, classes, train_prev, embedddings, labels)
    # log.info(f"[{h_info.name}@{d_info.name}] embeddings saved")
    p_info.dump(logits=dict(V=logits["validation"], U=logits["test"]))
    # log.info(f"[{h_info.name}@{d_info.name}] logits saved")


if __name__ == "__main__":
    if (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or "TOKENIZERS_PARALLELISM" not in os.environ
        or "HF_HUB_CACHE" not in os.environ
    ):
        raise ValueError("Missing env variables")

    parser = ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain existing models")
    parser.add_argument("--ignore-existing", dest="ignore_exist", action="store_true", help="Retrain existing models")
    parser.add_argument("--dry-run", action="store_true", help="Train the model without saving outputs")
    parser_args = parser.parse_args()

    # log.info("-" * 31 + "  start  " + "-" * 31)
    for d_info, h_info in gen_config():
        try:
            pretrain(d_info, h_info, parser_args)
        except Exception as e:
            # log.error(e)
            print_exception(e)
    # log.info("-" * 32 + "  end  " + "-" * 32)

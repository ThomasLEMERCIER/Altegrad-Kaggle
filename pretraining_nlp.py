# Standard library imports
import os
import math
import argparse
import os.path as osp
from yaml import safe_load

# Related third party imports
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

# Local application/library specific imports
from src.constants import *
from src.nlp_pretraining import get_corpus_generator, masked_training_dataset

def train_tokenizer(model_name, saving_path):
    train_set = get_corpus_generator(ROOT_DATA, "train")

    base_tokenizer = AutoTokenizer.from_pretrained(model_name)

    new_tokenizer = base_tokenizer.train_new_from_iterator(
        train_set, base_tokenizer.vocab_size
    )

    new_tokenizer.save_pretrained(saving_path)

    return new_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="pretraining.yaml", help="Name of config file"
    )

    args = parser.parse_args()
    config_path = osp.join("configs", args.config)
    config = safe_load(open(config_path, "r"))

    model_name = config["model_name"]
    batch_size = config["batch_size"]
    nb_epochs = config["nb_epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    mask_prob = config["mask_prob"]

    saving_folder = osp.join(CHECKPOINT_FOLDER, "pretraining", model_name)
    os.makedirs(saving_folder, exist_ok=True)

    tokenizer_path = osp.join(saving_folder, "tokenizer")
    if osp.exists(tokenizer_path):
        print("Tokenizer already trained")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Training tokenizer")
        tokenizer = train_tokenizer(model_name, tokenizer_path)

    masked_train_dataset = masked_training_dataset(ROOT_DATA, "train", tokenizer)
    masked_val_dataset = masked_training_dataset(ROOT_DATA, "val", tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=saving_folder,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=nb_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=len(masked_train_dataset) // batch_size,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=mask_prob
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=masked_train_dataset,
        eval_dataset=masked_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

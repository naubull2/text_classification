import os
import logging
import argparse
import numpy as np
import datasets
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 0.5B LLM for binary classification with DeepSpeed ZeRO-2 and early stopping."
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./qwen2.5_mrpc",
                        help="Directory to store the final model")
    parser.add_argument("--dataset_name", type=str, default="glue",
                        help="Name of the dataset to use (from Hugging Face Datasets)")
    parser.add_argument("--dataset_config", type=str, default="mrpc",
                        help="Dataset configuration (e.g., 'mrpc' for GLUE)")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--deepspeed_config", type=str, default="./ds_zero2.json",
                        help="Path to the DeepSpeed configuration file")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Steps between logging events")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Steps between model saves")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluation calls with no improvement before stopping early")
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_length):
    # For MRPC, tokenize sentence pairs without fixed padding.
    # Truncation is applied to ensure sequences do not exceed max_length.
    return tokenizer(examples["sentence1"], examples["sentence2"],
                     truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

def main():
    args = parse_args()

    # Set up logging.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
    
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Ensure a pad token is defined. For many decoder-only models, using EOS as pad is common.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    
    logger.info("Tokenizing dataset...")
    tokenize_fn = lambda examples: preprocess_function(examples, tokenizer, args.max_length)
    encoded_dataset = dataset.map(tokenize_fn, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True, # use memory saving if needed.
        deepspeed=args.deepspeed_config,
        report_to="none",  # Disable logging to external systems (e.g., WandB)
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Monitor accuracy for best model saving
        greater_is_better=True,
        save_total_limit=3  # Keep only a few checkpoints to save disk space
    )
    
    # Set up Trainer with early stopping callback.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete. Best model saved in %s", args.output_dir)


if __name__ == "__main__":
    main()


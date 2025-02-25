import os
import logging
import argparse
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import torch


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 0.5B LLM for binary classification with DeepSpeed ZeRO-2")
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct", 
                        help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./qwen2.5_mrpc",
                        help="Where to store the final model")
    parser.add_argument("--dataset_name", type=str, default="glue",
                        help="The name of the dataset to use (from Hugging Face Datasets)")
    parser.add_argument("--dataset_config", type=str, default="mrpc",
                        help="The configuration name of the dataset to use")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=32,
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
                        help="Number of steps between logging events")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between model saves")
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_length):
    # For MRPC, we tokenize sentence pairs.
    return tokenizer(examples["sentence1"], examples["sentence2"], 
                     truncation=True, padding="max_length", max_length=max_length)

def main():
    args = parse_args()
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
    
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    logger.info("Tokenizing dataset...")
    tokenize_fn = lambda examples: preprocess_function(examples, tokenizer, args.max_length)
    encoded_dataset = dataset.map(tokenize_fn, batched=True)
    

    # Set up training arguments
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
        fp16=True,
        deepspeed=args.deepspeed_config,
        report_to="none"  # disable external logging (e.g., wandb)
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()


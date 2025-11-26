#!/usr/bin/env python3
"""
Train DistilGPT-2 on anime chat data (JSONL format)
Fixed for Transformers 4.30.2 compatibility
"""

import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl_data(jsonl_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prompt_pairs(data: List[Dict]) -> Tuple[List[str], int]:
    """
    Create prompt + target pairs for causal language modeling.
    Format: [model_input] + [model_target] + <END>
    """
    prompts = []
    for item in data:
        model_input = item.get("model_input", "")
        model_target = item.get("model_target", "")
        
        if model_input and model_target:
            full_prompt = f"{model_input} {model_target} <END>"
            prompts.append(full_prompt)
    
    return prompts, len(prompts)


def train_gpt2_on_jsonl():
    """Fine-tune DistilGPT-2 on anime chat JSONL data."""
    
    logger.info("Starting DistilGPT-2 fine-tuning on JSONL data...")
    
    # ===== DATA LOADING =====
    jsonl_path = "C:/Users/Administrator/Desktop/anime-chatbot/checkpoint 4/anime-chatbot/backend/data/conversations/anime_chatbot_5000.jsonl"
    logger.info(f"Loading data from: {jsonl_path}")
    
    if not Path(jsonl_path).exists():
        logger.error(f"File not found: {jsonl_path}")
        return
    
    data = load_jsonl_data(jsonl_path)
    logger.info(f"Loaded {len(data)} training examples from {jsonl_path}")
    
    # Create prompt pairs
    logger.info("Creating prompt pairs...")
    prompts, total_prompts = create_prompt_pairs(data)
    logger.info(f"Created {total_prompts} prompts")
    
    # Split into train/eval
    split_idx = int(0.9 * len(prompts))
    train_prompts = prompts[:split_idx]
    eval_prompts = prompts[split_idx:]
    logger.info(f"Train: {len(train_prompts)} examples, Eval: {len(eval_prompts)} examples")
    
    # ===== MODEL LOADING =====
    logger.info("Loading distilgpt2...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== DATASET PREPARATION =====
    logger.info("Tokenizing datasets...")
    
    def tokenize_function(examples):
        """Tokenize function for dataset mapping."""
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_prompts})
    eval_dataset = Dataset.from_dict({"text": eval_prompts})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # ===== TRAINING SETUP =====
    logger.info("Setting up training arguments...")
    
    # FIXED: Changed eval_strategy to evaluation_strategy
    training_args = TrainingArguments(
        output_dir="backend/models/distilgpt2_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="backend/logs/distilgpt2",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        # FIXED: Use evaluation_strategy instead of eval_strategy
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        seed=42,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # ===== TRAINING =====
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # ===== SAVE MODEL =====
    logger.info("Saving fine-tuned model...")
    output_dir = "C:/Users/Administrator/Desktop/anime-chatbot/checkpoint 4/anime-chatbot/backend/models/distilgpt2_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    train_gpt2_on_jsonl()

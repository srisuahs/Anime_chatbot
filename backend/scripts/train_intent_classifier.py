#!/usr/bin/env python3
"""
Train DistilBERT-based Intent Classifier (FIXED FOR 1000-EXAMPLE DATASET)

✅ Works with new JSON format: [{"text": "...", "label": "..."}, ...]
✅ Handles both list and dict formats automatically
✅ 10 intents, 1000 total examples
✅ Fixed for Transformers 4.30.x compatibility

Run: python backend/scripts/train_intent_classifier.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Paths =====
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_training_data_1000.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "intent_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Data path: {DATA_PATH}")
logger.info(f"Output dir: {OUTPUT_DIR}")


def load_training_data() -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Load training data from JSON file.
    
    Handles TWO formats:
    1. List format (NEW): [{"text": "...", "label": "..."}, ...]
    2. Dict format (OLD): {"intents": [{"intent": "...", "examples": [...]}, ...]}
    
    Returns: texts, labels, label_map
    """
    logger.info(f"Loading data from {DATA_PATH}...")
    
    if not DATA_PATH.exists():
        logger.error(f"❌ File not found: {DATA_PATH}")
        sys.exit(1)
    
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {e}")
        sys.exit(1)
    
    texts: List[str] = []
    labels: List[int] = []
    label_map: Dict[str, int] = {}
    
    # ===== DETECT FORMAT =====
    if isinstance(data, list):
        # NEW FORMAT: [{"text": "...", "label": "..."}, ...]
        logger.info("Detected NEW format (list)")
        
        # Build label map from all labels in data
        unique_labels = set()
        for item in data:
            if isinstance(item, dict) and "label" in item:
                unique_labels.add(item["label"])
        
        label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        logger.info(f"Label map: {label_map}")
        
        # Extract texts and labels
        for item in data:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item: {item}")
                continue
            
            text = item.get("text")
            label = item.get("label")
            
            if text is None or label is None:
                logger.warning(f"Skipping incomplete item: {item}")
                continue
            
            texts.append(text)
            labels.append(label_map[label])
    
    elif isinstance(data, dict):
        # OLD FORMAT: {"intents": [{"intent": "...", "examples": [...]}, ...]}
        logger.info("Detected OLD format (dict)")
        
        intents = data.get("intents", [])
        if not intents:
            logger.error("❌ No intents found in data")
            sys.exit(1)
        
        # Build label map
        label_map = {intent.get("intent"): idx for idx, intent in enumerate(intents)}
        logger.info(f"Label map: {label_map}")
        
        # Extract texts and labels
        for intent_entry in intents:
            intent_name = intent_entry.get("intent")
            if intent_name not in label_map:
                label_map[intent_name] = len(label_map)
            
            idx = label_map[intent_name]
            for example in intent_entry.get("examples", []):
                if isinstance(example, str):
                    texts.append(example)
                    labels.append(idx)
    else:
        logger.error(f"❌ Unexpected data format: {type(data)}")
        sys.exit(1)
    
    # Validation
    if not texts or not labels:
        logger.error("❌ No training data loaded")
        sys.exit(1)
    
    if len(texts) != len(labels):
        logger.error(f"❌ Mismatch: {len(texts)} texts vs {len(labels)} labels")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(texts)} training examples for {len(label_map)} intents")
    
    # Print statistics
    for label_name, label_id in sorted(label_map.items(), key=lambda x: x[1]):
        count = labels.count(label_id)
        logger.info(f"   {label_name}: {count} examples")
    
    return texts, labels, label_map


def create_dataset(texts: List[str], labels: List[int]) -> Tuple[Dataset, Dataset]:
    """Create HuggingFace datasets and split into train/eval."""
    logger.info("Creating datasets...")
    
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    train_eval_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = train_eval_split["train"]
    eval_ds = train_eval_split["test"]
    
    logger.info(f"   Train size: {len(train_ds)}")
    logger.info(f"   Eval size: {len(eval_ds)}")
    
    return train_ds, eval_ds


def tokenize_function(examples, tokenizer):
    """Tokenize examples using the provided tokenizer."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def prepare_tokenized_datasets(
    train_ds: Dataset, eval_ds: Dataset, tokenizer
) -> Tuple[Dataset, Dataset]:
    """Tokenize both train and eval datasets."""
    logger.info("Tokenizing datasets...")
    
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        desc="Tokenizing train",
    )
    
    eval_ds = eval_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        desc="Tokenizing eval",
    )
    
    return train_ds, eval_ds


def train_classifier():
    """Main training function for intent classifier."""
    logger.info("=" * 70)
    logger.info("🚀 TRAINING DistilBERT Intent Classifier (1000 Examples)")
    logger.info("=" * 70)
    
    # ===== DATA LOADING =====
    logger.info("\n1️⃣ Loading training data...")
    texts, labels, label_map = load_training_data()
    
    # Save label map for later reference
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"✅ Label map saved to {label_map_path}")
    
    # ===== CREATE DATASETS =====
    logger.info("\n2️⃣ Creating datasets...")
    train_ds, eval_ds = create_dataset(texts, labels)
    
    # ===== LOAD MODEL & TOKENIZER =====
    logger.info("\n3️⃣ Loading model and tokenizer...")
    model_name = "distilbert-base-uncased"
    
    logger.info(f"   Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"   Loading model from {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_map),
    )
    logger.info(f"✅ Model loaded with {len(label_map)} labels")
    
    # ===== TOKENIZE DATASETS =====
    logger.info("\n4️⃣ Tokenizing datasets...")
    train_ds, eval_ds = prepare_tokenized_datasets(train_ds, eval_ds, tokenizer)
    
    # ===== TRAINING ARGUMENTS =====
    logger.info("\n5️⃣ Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,  # Increased for 1000 examples
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        seed=42,
    )
    logger.info("✅ Training arguments configured")
    
    # ===== CREATE TRAINER =====
    logger.info("\n6️⃣ Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    logger.info("✅ Trainer created")
    
    # ===== START TRAINING =====
    logger.info("\n7️⃣ Starting training...")
    logger.info("=" * 70)
    trainer.train()
    logger.info("=" * 70)
    
    # ===== SAVE MODEL =====
    logger.info("\n8️⃣ Saving model...")
    logger.info(f"   Saving to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("✅ Model and tokenizer saved")
    
    # ===== DONE =====
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\n📁 Model saved to: {OUTPUT_DIR}")
    logger.info(f"🏷️  Label map: {label_map}")
    logger.info(f"📊 Total examples: {len(texts)}")
    logger.info(f"🎯 Intents: {len(label_map)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        train_classifier()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

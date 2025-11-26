#!/usr/bin/env python3
"""
Fast Fine-Tuning Script for Phi-3-mini with LoRA (Parameter-Efficient)
Uses existing model in .anime_chatbot_cache and anime_chatbot_5000.jsonl dataset
"""

import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

# ===== PATHS =====
HOME = Path.home()
MODEL_CACHE = HOME / ".anime_chatbot_cache"
TRAINING_DATA = Path("anime_chatbot_5000.jsonl")  # Your 5000 training examples
OUTPUT_DIR = HOME / "phi3_finetuned_lora"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🚀 PHI-3 FAST FINE-TUNING WITH LORA")
print("=" * 70)
print(f"Model cache: {MODEL_CACHE}")
print(f"Training data: {TRAINING_DATA}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)

# ===== DEVICE SETUP =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ===== LOAD MODEL WITH 4-BIT QUANTIZATION =====
print("\n📦 Loading Phi-3-mini from cache...")

# 4-bit quantization config (saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model from cache
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir=str(MODEL_CACHE),
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # Use eager attention (more stable)
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    cache_dir=str(MODEL_CACHE),
    trust_remote_code=True
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("✅ Model loaded")

# ===== PREPARE MODEL FOR LORA =====
print("\n🔧 Preparing model for LoRA fine-tuning...")

# Prepare model for k-bit training (required for LoRA + quantization)
model = prepare_model_for_kbit_training(model)

# LoRA configuration (ONLY trains ~10%) of parameters!)
lora_config = LoraConfig(
    r=200,                          # LoRA rank (higher = more parameters, slower)
    lora_alpha=400,                 # LoRA scaling
    target_modules=[               # Which layers to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA configured")
print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"   Total params: {total_params:,}")

# ===== LOAD DATASET =====
print(f"\n📂 Loading training data from {TRAINING_DATA}...")

dataset = load_dataset("json", data_files=str(TRAINING_DATA), split="train")
print(f"✅ Loaded {len(dataset)} training examples")

# ===== TOKENIZATION FUNCTION =====
def tokenize_function(examples):
    """Tokenize the conversations for training"""
    # Format: <|system|>...<|user|>...<|assistant|>...
    conversations = []
    for messages in examples["messages"]:
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                conversation += f"<|system|>\n{content}<|end|>\n"
            elif role == "user":
                conversation += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                conversation += f"<|assistant|>\n{content}<|end|>\n"
        conversations.append(conversation)
    
    # Tokenize
    tokenized = tokenizer(
        conversations,
        truncation=True,
        max_length=512,  # Shorter context for faster training
        padding="max_length",
        return_tensors="pt"
    )
    
    # Set labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

print("\n🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)
print("✅ Dataset tokenized")

# ===== TRAINING ARGUMENTS =====
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    
    # Training speed settings (FAST!)
    num_train_epochs=3,              # 3 epochs (adjust if needed)
    per_device_train_batch_size=4,   # Batch size (increase if you have more VRAM)
    gradient_accumulation_steps=4,   # Effective batch size = 4*4 = 16
    
    # Optimizer settings
    learning_rate=2e-4,              # LoRA typically uses higher LR
    warmup_steps=100,
    weight_decay=0.01,
    
    # Logging
    logging_steps=50,
    logging_dir=str(OUTPUT_DIR / "logs"),
    
    # Saving
    save_steps=500,
    save_total_limit=2,              # Keep only 2 checkpoints to save disk space
    
    # Performance
    fp16=True if device == "cuda" else False,  # Mixed precision (faster on GPU)
    dataloader_num_workers=4,
    remove_unused_columns=False,
    
    # Misc
    report_to="none",                # Don't use wandb/tensorboard
    push_to_hub=False,
)

# ===== DATA COLLATOR =====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (not masked LM)
)

# ===== TRAINER =====
print("\n🏋️ Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("✅ Trainer ready")

# ===== TRAINING =====
print("\n" + "=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)
print(f"Training examples: {len(tokenized_dataset)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: ~{len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
print("=" * 70)

# Train!
trainer.train()

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)

# ===== SAVE MODEL =====
print("\n💾 Saving fine-tuned model...")

# Save LoRA adapter weights (only the trained parameters!)
model.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))

print(f"✅ Model saved to: {OUTPUT_DIR / 'lora_adapter'}")

# ===== USAGE INSTRUCTIONS =====
print("\n" + "=" * 70)
print("📋 HOW TO USE THE FINE-TUNED MODEL")
print("=" * 70)
print(f"""
To load and use the fine-tuned model in your chatbot:

1. Load base model:
   model = AutoModelForCausalLM.from_pretrained(
       "microsoft/Phi-3-mini-4k-instruct",
       cache_dir="{MODEL_CACHE}",
       quantization_config=bnb_config,
       device_map="auto"
   )

2. Load LoRA adapter:
   from peft import PeftModel
   model = PeftModel.from_pretrained(
       model,
       "{OUTPUT_DIR / 'lora_adapter'}"
   )

3. Use as normal:
   response = model.generate(...)
""")

print("\n" + "=" * 70)
print("🎉 DONE!")
print("=" * 70)

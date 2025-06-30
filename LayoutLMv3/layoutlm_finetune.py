#!/usr/bin/env python3
"""
LayoutLMv3 Fine-Tuning Module

This module implements LayoutLMv3 fine-tuning for bank statement PDF processing
with MacBook M1 compatibility and robust error handling.

Date: June 25, 2025
"""

import os
import json
import logging
import time
import torch
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase
)
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_parser.utils import load_config, get_project_root
from pdf_parser.logging_enhanced import EnhancedLogger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for LayoutLMv3 fine-tuning"""
    base_model: str = "microsoft/layoutlmv3-base"
    output_dir: str = "models/layoutlm_finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # M1 compatible
    per_device_eval_batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    gradient_accumulation_steps: int = 4  # Effective batch size = 4
    fp16: bool = False  # Disable FP16 for M1 compatibility
    dataloader_num_workers: int = 0  # M1 compatibility
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3


@dataclass
class FineTuningResult:
    """Results from LayoutLMv3 fine-tuning"""
    success: bool
    model_path: Optional[str]
    training_time: float
    final_metrics: Dict[str, float]
    training_history: List[Dict[str, float]]
    error_message: Optional[str] = None


class BankStatementDataset(Dataset):
    """Dataset class for bank statement token classification"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        processor: Any,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        label2id: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of training samples
            processor: LayoutLMv3 processor
            tokenizer: LayoutLMv3 tokenizer
            max_length: Maximum sequence length
            label2id: Label to ID mapping
        """
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {
            "O": 0, "B-DATE": 1, "I-DATE": 2, "B-DESC": 3, "I-DESC": 4,
            "B-AMOUNT": 5, "I-AMOUNT": 6, "B-BALANCE": 7, "I-BALANCE": 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        logger.info(f"Dataset initialized with {len(data)} samples")
        logger.info(f"Label mapping: {self.label2id}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        sample = self.data[idx]
        
        try:
            # Extract sample data
            tokens = sample.get('tokens', [])
            labels = sample.get('ner_tags', sample.get('labels', []))  # Try ner_tags first, then labels
            bbox = sample.get('bbox', [])
            
            # Handle missing labels - assign default O (outside) labels
            if not labels or len(labels) == 0:
                labels = [0] * len(tokens)  # All O labels
                logger.warning(f"Sample {idx} has no labels, using default O labels")
            
            # Ensure consistent lengths
            min_length = min(len(tokens), len(bbox))
            if labels:
                min_length = min(min_length, len(labels))
            
            tokens = tokens[:min_length]
            labels = labels[:min_length] if labels else [0] * min_length
            bbox = bbox[:min_length]
            
            # Handle token format - convert tokens list to words for LayoutLMv3
            # LayoutLMv3 expects word-level text, not sub-word tokens
            if isinstance(tokens[0], str) and len(tokens[0]) < 5:  # Likely sub-word tokens
                # Convert tokens back to text and then to words
                text = ''.join(tokens).replace('Ġ', ' ').strip()
                words = text.split()
            else:
                # Assume tokens are already words
                words = [str(token) for token in tokens]
            
            # Aggregate bounding boxes and labels at word level  
            # Use a more sophisticated approach to preserve meaningful labels
            word_boxes = []
            word_labels = []
            
            if len(words) > 0 and len(tokens) > 0:
                tokens_per_word = len(tokens) // len(words)
                remaining_tokens = len(tokens) % len(words)
                
                token_idx = 0
                for i, word in enumerate(words):
                    # Calculate how many tokens belong to this word
                    num_tokens = tokens_per_word + (1 if i < remaining_tokens else 0)
                    word_start_idx = token_idx
                    word_end_idx = min(token_idx + num_tokens, len(tokens))
                    
                    if word_start_idx < len(bbox):
                        # Use the first bounding box for this word
                        word_boxes.append(bbox[word_start_idx])
                    
                    if word_start_idx < len(labels):
                        # Find the most meaningful label in this word's token range
                        word_token_labels = labels[word_start_idx:word_end_idx]
                        # Prioritize non-O labels (non-zero)
                        non_zero_labels = [l for l in word_token_labels if l != 0]
                        if non_zero_labels:
                            word_labels.append(non_zero_labels[0])  # Use first non-O label
                        else:
                            word_labels.append(word_token_labels[0] if word_token_labels else 0)
                    
                    token_idx = word_end_idx
            
            # Ensure we have boxes and labels for all words
            while len(word_boxes) < len(words) and bbox:
                word_boxes.append(bbox[0])  # Use first box as default
            while len(word_labels) < len(words):
                word_labels.append(0)  # Use O label as default
            
            # Create dummy image (LayoutLMv3 requires image input)
            dummy_image = Image.new('RGB', (224, 224), color='white')
            
            # Tokenize with processor
            encoding = self.processor(
                dummy_image,
                words,  # Use words instead of raw text
                boxes=word_boxes if word_boxes else None,
                word_labels=word_labels,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            )
            
            # Remove batch dimension and ensure correct types
            for key in encoding:
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].squeeze(0)
                    if key in ['input_ids', 'bbox', 'attention_mask', 'labels']:
                        encoding[key] = encoding[key].long()
            
            return encoding
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return minimal valid sample
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'bbox': torch.zeros((self.max_length, 4), dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }


class LayoutLMv3FineTuner:
    """LayoutLMv3 fine-tuning engine"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize fine-tuning engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.project_root = get_project_root()
        
        # Initialize enhanced logging
        self.enhanced_logger = EnhancedLogger()
        
        # Fine-tuning configuration
        self.ft_config = FineTuningConfig()
        
        # Update from config file if available
        model_config = self.config.get('models', {}).get('layoutlmv3', {})
        if 'model_name' in model_config:
            self.ft_config.base_model = model_config['model_name']
        
        # Device setup for M1 compatibility
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using Apple M1 MPS acceleration")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU (MPS not available)")
        
        # Model components
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        
        logger.info(f"LayoutLMv3FineTuner initialized")
        logger.info(f"Base model: {self.ft_config.base_model}")
        logger.info(f"Output directory: {self.ft_config.output_dir}")
    
    def load_training_data(self, train_path: str, val_path: str = None, test_path: str = None) -> Tuple[List, List, List]:
        """
        Load training, validation, and test data.
        
        Args:
            train_path: Path to training data JSON
            val_path: Path to validation data JSON
            test_path: Path to test data JSON
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Loading training data...")
        
        # Load training data
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        logger.info(f"Loaded {len(train_data)} training samples")
        
        # Load validation data
        val_data = []
        if val_path and os.path.exists(val_path):
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            logger.info(f"Loaded {len(val_data)} validation samples")
        
        # Load test data
        test_data = []
        if test_path and os.path.exists(test_path):
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            logger.info(f"Loaded {len(test_data)} test samples")
        
        # Load label mapping
        labels_path = Path(train_path).parent / "labels.json"
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                label_info = json.load(f)
                self.label2id = label_info.get('label2id', {})
                self.id2label = label_info.get('id2label', {})
        else:
            # Default label mapping
            self.label2id = {
                "O": 0, "B-DATE": 1, "I-DATE": 2, "B-DESC": 3, "I-DESC": 4,
                "B-AMOUNT": 5, "I-AMOUNT": 6, "B-BALANCE": 7, "I-BALANCE": 8
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
        
        logger.info(f"Label mapping: {self.label2id}")
        return train_data, val_data, test_data
    
    def prepare_model_and_processor(self):
        """Prepare LayoutLMv3 model and processor for fine-tuning"""
        logger.info(f"Loading LayoutLMv3 model: {self.ft_config.base_model}")
        
        try:
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(self.ft_config.base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.ft_config.base_model)
            
            # Configure processor to disable OCR when using bounding boxes
            if hasattr(self.processor, 'image_processor'):
                self.processor.image_processor.apply_ocr = False
            
            # Load model for token classification
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.ft_config.base_model,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True
            )
            
            # Move to device
            self.model.to(self.device)
            
            logger.info("Model and processor loaded successfully")
            logger.info(f"Model has {self.model.num_parameters():,} parameters")
            
        except Exception as e:
            error_msg = f"Failed to load model and processor: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_datasets(self, train_data: List, val_data: List = None) -> Tuple:
        """
        Create PyTorch datasets for training.
        
        Args:
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info("Creating datasets...")
        
        # Create training dataset
        train_dataset = BankStatementDataset(
            data=train_data,
            processor=self.processor,
            tokenizer=self.tokenizer,
            max_length=self.ft_config.max_length,
            label2id=self.label2id
        )
        
        # Create validation dataset
        val_dataset = None
        if val_data:
            val_dataset = BankStatementDataset(
                data=val_data,
                processor=self.processor,
                tokenizer=self.tokenizer,
                max_length=self.ft_config.max_length,
                label2id=self.label2id
            )
        
        logger.info(f"Created training dataset with {len(train_dataset)} samples")
        if val_dataset:
            logger.info(f"Created validation dataset with {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Remove padding (-100 labels)
        true_labels = []
        pred_labels = []
        
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    true_labels.append(labels[i, j])
                    pred_labels.append(predictions[i, j])
        
        # Calculate metrics
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        
        return {
            "f1": f1,
            "accuracy": np.mean(np.array(true_labels) == np.array(pred_labels))
        }
    
    def fine_tune(
        self,
        train_data: List,
        val_data: List = None,
        resume_from_checkpoint: str = None
    ) -> FineTuningResult:
        """
        Fine-tune LayoutLMv3 model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            FineTuningResult with training results
        """
        start_time = time.time()
        
        try:
            logger.info("Starting LayoutLMv3 fine-tuning...")
            
            # Prepare model and processor
            self.prepare_model_and_processor()
            
            # Create datasets
            train_dataset, val_dataset = self.create_datasets(train_data, val_data)
            
            # Create output directory
            os.makedirs(self.ft_config.output_dir, exist_ok=True)
            
            # Training arguments optimized for M1
            training_args = TrainingArguments(
                output_dir=self.ft_config.output_dir,
                num_train_epochs=self.ft_config.num_train_epochs,
                per_device_train_batch_size=self.ft_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.ft_config.per_device_eval_batch_size,
                learning_rate=self.ft_config.learning_rate,
                weight_decay=self.ft_config.weight_decay,
                warmup_ratio=self.ft_config.warmup_ratio,
                logging_steps=self.ft_config.logging_steps,
                save_steps=self.ft_config.save_steps,
                eval_steps=self.ft_config.eval_steps if val_dataset else None,
                eval_strategy="steps" if val_dataset else "no",
                save_total_limit=self.ft_config.save_total_limit,
                load_best_model_at_end=self.ft_config.load_best_model_at_end and val_dataset is not None,
                metric_for_best_model=self.ft_config.metric_for_best_model,
                greater_is_better=self.ft_config.greater_is_better,
                gradient_accumulation_steps=self.ft_config.gradient_accumulation_steps,
                fp16=self.ft_config.fp16,
                dataloader_num_workers=self.ft_config.dataloader_num_workers,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to=None  # Disable wandb/tensorboard for simplicity
            )
            
            # Data collator
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding=True,
                max_length=self.ft_config.max_length
            )
            
            # Callbacks
            callbacks = []
            if val_dataset and self.ft_config.early_stopping_patience > 0:
                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=self.ft_config.early_stopping_patience)
                )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics if val_dataset else None,
                callbacks=callbacks
            )
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.ft_config.output_dir)
            self.processor.save_pretrained(self.ft_config.output_dir)
            
            # Save training configuration
            try:
                config_path = os.path.join(self.ft_config.output_dir, "training_config.json")
                with open(config_path, 'w') as f:
                    json.dump({
                        "base_model": self.ft_config.base_model,
                        "num_train_epochs": self.ft_config.num_train_epochs,
                        "batch_size": self.ft_config.per_device_train_batch_size,
                        "learning_rate": self.ft_config.learning_rate,
                        "label2id": self.label2id,
                        "id2label": self.id2label,
                        "training_samples": len(train_data),
                        "validation_samples": len(val_data) if val_data else 0,
                        "training_time": time.time() - start_time
                    }, f, indent=2)
                logger.info(f"Training configuration saved to {config_path}")
            except Exception as e:
                logger.warning(f"Failed to save training configuration: {e}")
            
            # Final evaluation
            final_metrics = {}
            if val_dataset:
                eval_result = trainer.evaluate()
                final_metrics = eval_result
                logger.info(f"Final evaluation metrics: {final_metrics}")
            
            training_time = time.time() - start_time
            logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
            logger.info(f"Model saved to: {self.ft_config.output_dir}")
            
            # Log training completion
            try:
                self.enhanced_logger.log_model_usage(
                    model_name="layoutlmv3-finetuned",
                    confidence_score=final_metrics.get('eval_f1', 0.0),
                    processing_time=training_time,
                    input_type="training",
                    tokens_processed=len(train_data),
                    routing_reason="fine_tuning_completion"
                )
            except Exception as e:
                logger.warning(f"Failed to log training completion: {e}")
            
            return FineTuningResult(
                success=True,
                model_path=self.ft_config.output_dir,
                training_time=training_time,
                final_metrics=final_metrics,
                training_history=trainer.state.log_history,
                error_message=None
            )
            
        except Exception as e:
            error_msg = f"Fine-tuning failed: {str(e)}"
            logger.error(error_msg)
            
            # Log training failure
            try:
                self.enhanced_logger.log_validation_error(
                    error_type="fine_tuning_error",
                    error_message=error_msg,
                    row_data={"model": "layoutlmv3", "stage": "fine_tuning"}
                )
            except Exception as log_error:
                logger.warning(f"Failed to log training failure: {log_error}")
            
            return FineTuningResult(
                success=False,
                model_path=None,
                training_time=time.time() - start_time,
                final_metrics={},
                training_history=[],
                error_message=error_msg
            )


def fine_tune_layoutlmv3(
    train_data_path: str = "data/training/layoutlm/train.json",
    val_data_path: str = "data/training/layoutlm/validation.json",
    test_data_path: str = "data/training/layoutlm/test.json",
    config_path: str = "config/config.yaml",
    resume_from_checkpoint: str = None
) -> FineTuningResult:
    """
    Fine-tune LayoutLMv3 model on bank statement data.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        test_data_path: Path to test data
        config_path: Path to configuration file
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        FineTuningResult with training results
    """
    fine_tuner = LayoutLMv3FineTuner(config_path)
    
    # Load data
    train_data, val_data, test_data = fine_tuner.load_training_data(
        train_data_path, val_data_path, test_data_path
    )
    
    # Start fine-tuning
    result = fine_tuner.fine_tune(train_data, val_data, resume_from_checkpoint)
    
    return result


if __name__ == "__main__":
    # Example usage
    result = fine_tune_layoutlmv3()
    
    if result.success:
        print(f"✅ Fine-tuning completed successfully!")
        print(f"Model saved to: {result.model_path}")
        print(f"Training time: {result.training_time:.2f} seconds")
        print(f"Final metrics: {result.final_metrics}")
    else:
        print(f"❌ Fine-tuning failed: {result.error_message}") 
#!/usr/bin/env python3
"""
Script to examine LayoutLMv3 training data structure
"""

import json
import sys
from pathlib import Path

def examine_training_data():
    """Examine the structure of LayoutLMv3 training data"""
    
    # Load training data
    train_file = Path("data/training/layoutlm/train.json")
    labels_file = Path("data/training/layoutlm/labels.json")
    
    print("=== LayoutLMv3 Training Data Analysis ===\n")
    
    # Load labels
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    print("Label Schema:")
    print(f"  Number of labels: {len(labels['label2id'])}")
    print("  Labels mapping:")
    for label, id_ in labels['label2id'].items():
        print(f"    {label} -> {id_}")
    print()
    
    # Load and examine training samples
    with open(train_file, 'r') as f:
        training_data = json.load(f)
    
    print(f"Training Data Overview:")
    print(f"  Total samples: {len(training_data)}")
    print()
    
    # Examine first few samples
    print("Sample Structure (first 3 samples):")
    for i, sample in enumerate(training_data[:3]):
        print(f"  Sample {i+1}:")
        print(f"    ID: {sample.get('id', 'N/A')}")
        print(f"    Tokens count: {len(sample.get('tokens', []))}")
        print(f"    Labels count: {len(sample.get('labels', []))}")
        print(f"    Bbox count: {len(sample.get('bbox', []))}")
        
        # Show first few tokens and labels
        tokens = sample.get('tokens', [])[:10]
        labels = sample.get('labels', [])[:10]
        print(f"    First 10 tokens: {tokens}")
        print(f"    First 10 labels: {labels}")
        print()
    
    # Check data consistency
    total_tokens = sum(len(sample.get('tokens', [])) for sample in training_data)
    total_labels = sum(len(sample.get('labels', [])) for sample in training_data)
    total_bbox = sum(len(sample.get('bbox', [])) for sample in training_data)
    
    print("Data Consistency Check:")
    print(f"  Total tokens across all samples: {total_tokens}")
    print(f"  Total labels across all samples: {total_labels}")
    print(f"  Total bounding boxes: {total_bbox}")
    print(f"  Tokens == Labels: {total_tokens == total_labels}")
    print(f"  Tokens == Bbox: {total_tokens == total_bbox}")
    print()
    
    # Analyze label distribution
    label_counts = {}
    for sample in training_data:
        for label in sample.get('labels', []):
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label Distribution:")
    for label_id, count in sorted(label_counts.items()):
        label_name = labels['id2label'].get(str(label_id), f'Unknown-{label_id}')
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        print(f"  {label_name} ({label_id}): {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    examine_training_data() 
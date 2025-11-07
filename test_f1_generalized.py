#!/usr/bin/env python3
"""
Test script for the generalized F1 score function.
"""

import torch
import sys
import os

# Add the parent directory to the path to import the metrics module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.accuracy import f1_score

def test_single_task():
    """Test single task F1 score (original behavior)"""
    print("Testing single task F1 score...")
    
    # Test case 1: Perfect predictions
    y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2])  # logits
    y_true = torch.tensor([1, 0, 1, 0])
    f1 = f1_score(y_pred, y_true, from_logits=True)
    print(f"Perfect predictions F1: {f1:.4f} (expected: 1.0)")
    
    # Test case 2: Random predictions
    y_pred = torch.tensor([0.3, 0.7, 0.4, 0.6])
    y_true = torch.tensor([1, 0, 1, 0])
    f1 = f1_score(y_pred, y_true, from_logits=True)
    print(f"Random predictions F1: {f1:.4f}")
    
    print()

def test_multi_task():
    """Test multi-task F1 score"""
    print("Testing multi-task F1 score...")
    
    # Test case 1: Multi-task with explicit multi_task=True
    y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.8], [0.8, 0.2], [0.2, 0.9]])  # [batch_size, num_tasks]
    y_true = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    f1 = f1_score(y_pred, y_true, from_logits=True, multi_task=True)
    print(f"Multi-task perfect predictions F1: {f1:.4f} (expected: 1.0)")
    
    # Test case 2: Multi-task with mixed performance
    y_pred = torch.tensor([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])  # Task 1: good, Task 2: good
    y_true = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    f1 = f1_score(y_pred, y_true, from_logits=True, multi_task=True)
    print(f"Multi-task mixed predictions F1: {f1:.4f}")
    
    # Test case 3: Auto-detect multi-task from shape
    y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.8], [0.8, 0.2], [0.2, 0.9]])
    y_true = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
    f1 = f1_score(y_pred, y_true, from_logits=True)  # Should auto-detect multi-task
    print(f"Auto-detected multi-task F1: {f1:.4f} (expected: 1.0)")
    
    print()

def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    # Test case 1: All zeros
    y_pred = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
    y_true = torch.tensor([[0, 0], [0, 0]])
    f1 = f1_score(y_pred, y_true, from_logits=True, multi_task=True)
    print(f"All zeros F1: {f1:.4f}")
    
    # Test case 2: All ones
    y_pred = torch.tensor([[0.9, 0.9], [0.8, 0.8]])
    y_true = torch.tensor([[1, 1], [1, 1]])
    f1 = f1_score(y_pred, y_true, from_logits=True, multi_task=True)
    print(f"All ones F1: {f1:.4f}")
    
    # Test case 3: Single sample multi-task
    y_pred = torch.tensor([[0.9, 0.1]])  # [1, 2]
    y_true = torch.tensor([[1, 0]])
    f1 = f1_score(y_pred, y_true, from_logits=True, multi_task=True)
    print(f"Single sample multi-task F1: {f1:.4f}")
    
    print()

def test_task_wise_computation():
    """Test that task-wise computation works correctly"""
    print("Testing task-wise computation...")
    
    # Create a case where task 1 is perfect and task 2 is terrible
    y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])  # Task 1: good, Task 2: bad
    y_true = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])  # Task 1: perfect, Task 2: wrong
    
    f1 = f1_score(y_pred, y_true, from_logits=True, multi_task=True)
    print(f"Mixed task performance F1: {f1:.4f}")
    
    # Manually compute expected F1 for each task
    # Task 1: pred=[1,0,1,0], true=[1,0,1,0] -> perfect -> F1=1.0
    # Task 2: pred=[0,1,0,1], true=[0,1,0,1] -> perfect -> F1=1.0
    # Average: (1.0 + 1.0) / 2 = 1.0
    print(f"Expected F1: 1.0 (both tasks perfect)")
    
    print()

if __name__ == "__main__":
    test_single_task()
    test_multi_task()
    test_edge_cases()
    test_task_wise_computation()
    print("All tests completed!")

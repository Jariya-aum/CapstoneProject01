# tests/test_metrics.py
import pytest

def test_calculate_accuracy():
    """ทดสอบการคำนวณ Accuracy"""
    # Confusion Matrix จำลอง: [[TN, FP], [FN, TP]]
    cm = [[1137, 3], [7, 262]]
    
    # คำนวณ Accuracy = (TP + TN) / Total
    tp, tn, fp, fn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    expected = 0.9929  # 99.29%
    assert round(accuracy, 4) == expected, f"Expected {expected}, got {round(accuracy, 4)}"
    print(f"✅ Accuracy Test Passed: {accuracy:.4f}")

def test_calculate_kappa():
    """ทดสอบการคำนวณ Kappa Coefficient"""
    cm = [[1137, 3], [7, 262]]
    
    tp, tn, fp, fn = cm[1][1], cm[0][0], cm[0][1], cm[1][0]
    total = tp + tn + fp + fn
    
    # Observed agreement
    po = (tp + tn) / total
    
    # Expected agreement by chance
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2)
    
    # Kappa = (po - pe) / (1 - pe)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0
    
    assert kappa > 0.70, f"Kappa should be > 0.70, got {kappa:.4f}"
    print(f"✅ Kappa Test Passed: {kappa:.4f}")

def test_confusion_matrix_structure():
    """ทดสอบโครงสร้าง Confusion Matrix"""
    cm = [[1137, 3], [7, 262]]
    
    # เช็กว่ามี 2 แถว 2 คอลัมน์
    assert len(cm) == 2, "Confusion matrix should have 2 rows"
    assert all(len(row) == 2 for row in cm), "Each row should have 2 columns"
    
    # เช็กว่าเป็นตัวเลขบวกทั้งหมด
    assert all(val >= 0 for row in cm for val in row), "All values should be non-negative"
    print("✅ Confusion Matrix Structure Test Passed")
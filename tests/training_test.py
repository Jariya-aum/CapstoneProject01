# tests/training_test.py
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def test_train_test_split_ratio():
    """ทดสอบว่า train-test split เป็นอัตรา 70:30"""
    from sklearn.model_selection import train_test_split
    
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    assert len(X_train) == 700
    assert len(X_test) == 300
    print("✅ Train-test split ratio test passed")

def test_random_forest_prediction():
    """ทดสอบว่า Random Forest สามารถ predict ได้"""
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(30, 10)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    assert len(predictions) == 30
    assert all(p in [0, 1] for p in predictions)
    print("✅ Random Forest prediction test passed")

def test_svm_prediction():
    """ทดสอบว่า SVM สามารถ predict ได้"""
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(30, 10)
    
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    
    assert len(predictions) == 30
    assert all(p in [0, 1] for p in predictions)
    print("✅ SVM prediction test passed")

def test_model_accuracy_reasonable():
    """ทดสอบว่า model มี accuracy > 50% (ดีกว่าทายสุ่ม)"""
    # สร้างข้อมูลที่แยกได้ง่าย
    np.random.seed(42)
    X_train = np.vstack([
        np.random.randn(50, 2) + [2, 2],  # Class 0
        np.random.randn(50, 2) + [-2, -2]  # Class 1
    ])
    y_train = np.array([0]*50 + [1]*50)
    
    X_test = np.vstack([
        np.random.randn(15, 2) + [2, 2],
        np.random.randn(15, 2) + [-2, -2]
    ])
    y_test = np.array([0]*15 + [1]*15)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.5, f"Accuracy {accuracy:.2f} should be > 0.5"
    print(f"✅ Model accuracy test passed: {accuracy:.4f}")
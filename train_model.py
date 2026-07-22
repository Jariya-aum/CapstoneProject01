# train_model.py
import ee
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import geemap
import joblib
import os

# Initialize Earth Engine
ee.Initialize(project='coastal-wares-439106-s9')

# ========== Helper Functions ==========

def extract_features_from_gee(aoi, crop_season_start, crop_season_end):
    """ดึงข้อมูล features จาก Google Earth Engine"""
    print("Fetching data from Google Earth Engine...")

    # Import จากไฟล์ที่แก้ไขแล้ว
    from ml_corn_function import (
        get_sentinel1_data,
        get_sentinel2_data,
        create_monthly_composites
    )

    # ดึงข้อมูล
    s1_raw = get_sentinel1_data(crop_season_start, crop_season_end, aoi)
    s2_raw = get_sentinel2_data(crop_season_start, crop_season_end, aoi)

    months = ee.List(list(range(5, 13))).cat(ee.List(list(range(1, 3))))
    s1_monthly = create_monthly_composites(s1_raw, s2_raw, months)

    # สร้าง composite
    s1_composite = ee.ImageCollection.fromImages(s1_monthly).toBands()

    # สุ่มตัวอย่าง
    sample = s1_composite.sample(
        region=aoi,
        scale=10,
        numPixels=5000,
        seed=42,
        geometries=True
    )

    return sample, s1_composite

def prepare_training_data(sample, s1_composite, aoi):
    """เตรียมข้อมูล training แบบ self-labeling"""
    print("Creating self-labels with KMeans...")

    # KMeans clustering
    clusterer = ee.Clusterer.wekaKMeans(5).train(sample)
    cluster_image = s1_composite.cluster(clusterer)

    # Self-labeling (ตามโค้ดเดิม)
    corn_cluster_id = 2
    corn_mask = cluster_image.eq(corn_cluster_id)
    non_corn_mask = cluster_image.neq(corn_cluster_id)

    corn_pixels = cluster_image.updateMask(corn_mask).rename('class').multiply(0).add(1)
    non_corn_pixels = cluster_image.updateMask(non_corn_mask).rename('class').multiply(0)
    label_image = corn_pixels.unmask().add(non_corn_pixels.unmask()).rename('class')

    # สร้าง training data
    training = s1_composite.addBands(label_image).sample(
        region=aoi,
        scale=10,
        numPixels=5000,
        seed=42
    )

    return training, cluster_image

def gee_to_pandas(training_fc):
    """แปลง Earth Engine FeatureCollection เป็น Pandas DataFrame"""
    print("Converting GEE data to Pandas...")

    # ดึงข้อมูลเป็น list
    features = training_fc.getInfo()['features']

    # แปลงเป็น DataFrame
    data = []
    for feature in features:
        props = feature['properties']
        data.append(props)

    df = pd.DataFrame(data)

    # แยก features และ labels
    if 'class' in df.columns:
        X = df.drop('class', axis=1)
        y = df['class']
        return X, y
    else:
        return df, None

# ========== Training Functions ==========

def train_random_forest(X_train, y_train, X_test, y_test, run_name="Random Forest"):
    """Train Random Forest + Log to MLflow"""

    with mlflow.start_run(run_name=run_name):
        print(f"\nTraining {run_name}...")

        # Hyperparameters
        n_estimators = 50
        min_samples_leaf = 5

        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        # Train model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        rf.fit(X_train, y_train)

        # Predict
        y_pred = rf.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("kappa", kappa)
        mlflow.log_metric("true_negative", int(cm[0][0]))
        mlflow.log_metric("false_positive", int(cm[0][1]))
        mlflow.log_metric("false_negative", int(cm[1][0]))
        mlflow.log_metric("true_positive", int(cm[1][1]))

        # Save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Non-Corn', 'Corn'],
                      yticklabels=['Non-Corn', 'Corn'])
        plt.title(f'{run_name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix_rf.png')
        mlflow.log_artifact('confusion_matrix_rf.png')
        plt.close()

        # Log model
        mlflow.sklearn.log_model(rf, "model")

        # Register model
        try:
            run = mlflow.active_run()
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, "RandomForest")
            print(f"Registered model: RandomForest (version {registered_model.version})")

            # Transition to Production stage
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.transition_model_version_stage(
                name="RandomForest",
                version=registered_model.version,
                stage="Production"
            )
            print(f"Set RandomForest v{registered_model.version} to Production stage")
        except Exception as e:
            print(f"Could not register model: {e}")

        print(f"{run_name} - Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}")

        return rf, accuracy, kappa

def train_svm(X_train, y_train, X_test, y_test, run_name="SVM"):
    """Train SVM + Log to MLflow"""

    with mlflow.start_run(run_name=run_name):
        print(f"\nTraining {run_name}...")

        # Hyperparameters
        kernel = 'rbf'
        C = 10
        gamma = 0.5

        # Log parameters
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("C", C)
        mlflow.log_param("gamma", gamma)

        # Train model
        svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        svm.fit(X_train, y_train)

        # Predict
        y_pred = svm.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("kappa", kappa)
        mlflow.log_metric("true_negative", int(cm[0][0]))
        mlflow.log_metric("false_positive", int(cm[0][1]))
        mlflow.log_metric("false_negative", int(cm[1][0]))
        mlflow.log_metric("true_positive", int(cm[1][1]))

        # Save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                      xticklabels=['Non-Corn', 'Corn'],
                      yticklabels=['Non-Corn', 'Corn'])
        plt.title(f'{run_name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix_svm.png')
        mlflow.log_artifact('confusion_matrix_svm.png')
        plt.close()

        # Log model
        mlflow.sklearn.log_model(svm, "model")

        # เพิ่มส่วนนี้: Register Model
        try:
            run = mlflow.active_run()
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, "SVM")
            print(f"Registered model: SVM (version {registered_model.version})")

            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.transition_model_version_stage(
                name="SVM",
                version=registered_model.version,
                stage="Staging"  # SVM ใช้ Staging
            )
            print(f"Set SVM v{registered_model.version} to Staging stage")
        except Exception as e:
            print(f"Could not register model: {e}")

        print(f"{run_name} - Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}")

        return svm, accuracy, kappa

# ========== Main Training Pipeline ==========

def main():
    """Main training pipeline"""

    # Set MLflow experiment
    mlflow.set_experiment("Detect-Corn-Classification")

    # Define AOI
    aoi = ee.Geometry.Rectangle([98.03724813148085, 18.32740564041369,
                                 98.36821126624648, 18.67642400421453])
    crop_season_start = '2024-05-01'
    crop_season_end = '2025-02-28'

    # Extract features
    sample, s1_composite = extract_features_from_gee(aoi, crop_season_start, crop_season_end)

    # Prepare training data
    training, cluster_image = prepare_training_data(sample, s1_composite, aoi)

    # Convert to Pandas
    X, y = gee_to_pandas(training)
    
    # เพิ่มส่วนนี้: บันทึก feature names
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print(f"Saved {len(X.columns)} feature names to models/feature_names.pkl")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nDataset size:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # Train models
    rf_model, rf_acc, rf_kappa = train_random_forest(X_train, y_train, X_test, y_test)
    svm_model, svm_acc, svm_kappa = train_svm(X_train, y_train, X_test, y_test)

    # Compare results
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Random Forest - Accuracy: {rf_acc:.4f}, Kappa: {rf_kappa:.4f}")
    print(f"SVM           - Accuracy: {svm_acc:.4f}, Kappa: {svm_kappa:.4f}")
    print("="*50)

    # Save best model
    best_model = rf_model if rf_acc > svm_acc else svm_model
    best_name = "RandomForest" if rf_acc > svm_acc else "SVM"

    joblib.dump(best_model, f'models/best_model_{best_name}.pkl')
    print(f"\nBest model ({best_name}) saved to models/")

    print("\nTraining completed! View results with: mlflow ui")

if __name__ == "__main__":
    main()
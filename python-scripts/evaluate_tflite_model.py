import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import json
import argparse

# -----------------------------
# 🧠 CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up from python-scripts to project root

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
BACKEND_MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
ASSETS_MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "images", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

IMG_SIZE = (224, 224)
TFLITE_MODEL_NAME = "outer_eye_mobilenetv2.tflite"

# Disease categories — must match training script
DISEASES = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"]

# Test split ratio (if using original dataset)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 📂 LOAD DATASET
# -----------------------------
def load_dataset():
    """Load images and labels from dataset directory."""
    print(f"📥 Loading dataset from: {DATASET_DIR}")
    images = []
    labels = []
    
    for disease in DISEASES:
        disease_path = os.path.join(DATASET_DIR, disease)
        if not os.path.exists(disease_path):
            print(f"⚠️ Missing folder for {disease} ({disease_path})")
            continue
        
        for img_file in tqdm(os.listdir(disease_path), desc=f"Loading {disease}"):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(disease_path, img_file)
                try:
                    img = image.load_img(img_path, target_size=IMG_SIZE)
                    img_array = image.img_to_array(img)
                    images.append(img_array)
                    labels.append(disease)
                except Exception as e:
                    print(f"⚠️ Could not read {img_file}: {e}")
    
    if len(images) == 0:
        raise ValueError("❌ No images loaded! Please check your dataset folder names and paths.")
    
    images = np.array(images, dtype=np.float32) / 255.0  # normalize to [0,1]
    labels = np.array(labels)
    
    print(f"✅ Loaded {len(images)} images")
    return images, labels

# -----------------------------
# 🧬 PREPARE TEST DATA
# -----------------------------
def prepare_test_data(images, labels, create_test_split=True):
    """
    Prepare test dataset.
    
    Args:
        images: All images
        labels: All labels
        create_test_split: If True, creates a test split. If False, uses full dataset.
    
    Returns:
        X_test: Test images
        y_test: Test labels (categorical)
        y_test_labels: Test labels (original string names)
        label_encoder: Fitted LabelEncoder
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
    if create_test_split:
        # Create a test split from the original data
        print("📊 Creating test split from dataset...")
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            images, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
    else:
        # Use all data as test (useful if you want to evaluate on full dataset)
        print("📊 Using full dataset for evaluation...")
        X_test = images
        y_test_encoded = y_encoded
    
    # Convert to categorical for model prediction
    y_test_categorical = to_categorical(y_test_encoded, num_classes=len(DISEASES))
    
    # Get original label names
    y_test_labels = le.inverse_transform(y_test_encoded)
    
    print(f"✅ Test set: {len(X_test)} images")
    print(f"📊 Class distribution:")
    unique, counts = np.unique(y_test_labels, return_counts=True)
    for disease, count in zip(unique, counts):
        print(f"   {disease}: {count}")
    
    return X_test, y_test_categorical, y_test_labels, le

# -----------------------------
# 🧠 LOAD TFLITE MODEL
# -----------------------------
def find_tflite_model(custom_model_path=None):
    """
    Find the TFLite model in backend or assets directory with fallback logic.
    
    Priority order:
    1. Custom path (if provided)
    2. Backend/models (primary - for server/backend use)
    3. Assets/images/models (fallback - for offline/mobile use)
    
    Args:
        custom_model_path: Optional custom path to model file. If provided, uses this instead.
    
    Returns:
        Path to the .tflite model file.
    """
    # If custom path provided, use it (highest priority)
    if custom_model_path:
        if os.path.exists(custom_model_path):
            print(f"✅ Using custom model path: {custom_model_path}")
            return custom_model_path
        else:
            raise FileNotFoundError(f"❌ Custom model path not found: {custom_model_path}")
    
    # Define paths
    backend_model_path = os.path.join(BACKEND_MODEL_DIR, TFLITE_MODEL_NAME)
    assets_model_path = os.path.join(ASSETS_MODEL_DIR, TFLITE_MODEL_NAME)
    
    # Check backend/models first (primary location)
    backend_exists = os.path.exists(backend_model_path)
    assets_exists = os.path.exists(assets_model_path)
    
    if backend_exists:
        print(f"✅ Found TFLite model in backend (primary): {backend_model_path}")
        return backend_model_path
    
    # Fallback to assets/images/models if backend is missing
    if assets_exists:
        print(f"⚠️  Backend TFLite model not found, using fallback from assets")
        print(f"✅ Found TFLite model in assets (fallback): {assets_model_path}")
        return assets_model_path
    
    # Neither .tflite model found - provide detailed error message
    error_msg = f"❌ TFLite model not found in any location!\n\n"
    error_msg += f"Checked locations (in priority order):\n"
    error_msg += f"  1. Backend (primary): {backend_model_path}\n"
    error_msg += f"  2. Assets (fallback): {assets_model_path}\n\n"
    error_msg += f"Please ensure the TFLite model file exists in one of these locations.\n"
    error_msg += f"Model name: {TFLITE_MODEL_NAME}\n"
    
    raise FileNotFoundError(error_msg)

def load_tflite_model(custom_model_path=None):
    """
    Load the TFLite model from backend or assets.
    
    Args:
        custom_model_path: Optional custom path to model file.
    
    Returns:
        TensorFlow Lite interpreter.
    """
    model_path = find_tflite_model(custom_model_path)
    
    print(f"📥 Loading TFLite model from: {model_path}")
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ TFLite model loaded successfully")
        print(f"📋 Input shape: {input_details[0]['shape']}")
        print(f"📋 Output shape: {output_details[0]['shape']}")
        print(f"📋 Input dtype: {input_details[0]['dtype']}")
        print(f"📋 Output dtype: {output_details[0]['dtype']}")
        
        return interpreter, input_details, output_details
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load TFLite model from {model_path}\n"
            f"Error: {str(e)}\n"
            f"Make sure the model file is a valid TensorFlow Lite .tflite model."
        )

# -----------------------------
# 🔮 RUN PREDICTIONS
# -----------------------------
def run_predictions_tflite(interpreter, input_details, output_details, X_test, batch_size=1):
    """
    Run predictions on test set using TFLite interpreter.
    
    Note: TFLite typically processes one image at a time, so batch_size is usually 1.
    """
    print(f"\n🔮 Running predictions on {len(X_test)} images using TFLite...")
    
    y_pred_proba = []
    
    # Process images one by one (TFLite doesn't support batching well)
    for i in tqdm(range(len(X_test)), desc="Predicting"):
        # Get input tensor
        input_data = X_test[i:i+1].astype(input_details[0]['dtype'])
        
        # Set the tensor to point to the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run the inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred_proba.append(output_data[0])
    
    y_pred_proba = np.array(y_pred_proba)
    
    # Get class predictions (argmax)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("✅ Predictions complete")
    return y_pred, y_pred_proba

# -----------------------------
# 📊 EXTRACT TRUE LABELS
# -----------------------------
def extract_true_labels(y_test_categorical):
    """Extract true labels from categorical format."""
    y_true = np.argmax(y_test_categorical, axis=1)
    return y_true

# -----------------------------
# 📈 GENERATE METRICS
# -----------------------------
def generate_metrics(y_true, y_pred, label_encoder, save_dir):
    """Generate comprehensive evaluation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save classification report to JSON
    report_path = os.path.join(save_dir, "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Saved classification report to: {report_path}")
    
    # Save detailed metrics to text file
    metrics_path = os.path.join(save_dir, "detailed_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TFLITE MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {TFLITE_MODEL_NAME}\n")
        f.write(f"Model Type: TensorFlow Lite\n")
        f.write(f"Test Set Size: {len(y_true)} images\n")
        f.write(f"Number of Classes: {len(class_names)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {int(support[i]):<10}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("MACRO AVERAGE\n")
        f.write("-" * 80 + "\n")
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        f.write(f"Precision: {macro_precision:.4f}\n")
        f.write(f"Recall: {macro_recall:.4f}\n")
        f.write(f"F1-Score: {macro_f1:.4f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("WEIGHTED AVERAGE\n")
        f.write("-" * 80 + "\n")
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        f.write(f"Precision: {weighted_precision:.4f}\n")
        f.write(f"Recall: {weighted_recall:.4f}\n")
        f.write(f"F1-Score: {weighted_f1:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'':<20}")
        for class_name in class_names:
            f.write(f"{class_name:<15}")
        f.write("\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<20}")
            for j in range(len(class_names)):
                f.write(f"{cm[i][j]:<15}")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFUSION MATRIX (Normalized - Percent)\n")
        f.write("=" * 80 + "\n\n")
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        f.write(f"{'':<20}")
        for class_name in class_names:
            f.write(f"{class_name:<15}")
        f.write("\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<20}")
            for j in range(len(class_names)):
                f.write(f"{cm_normalized[i][j]:<14.1f}%")
            f.write("\n")
    
    print(f"💾 Saved detailed metrics to: {metrics_path}")
    
    return report, cm, accuracy, precision, recall, f1, support

# -----------------------------
# 📊 VISUALIZE RESULTS
# -----------------------------
def visualize_results(cm, class_names, precision, recall, f1, save_dir):
    """Create visualization plots for evaluation results."""
    print("\n📊 Creating visualizations...")
    
    # ============================================
    # 1. PER-CLASS METRICS (Precision, Recall, F1)
    # ============================================
    print("📊 Creating per-class metrics plot...")
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    metrics_path = os.path.join(save_dir, "per_class_metrics.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved per-class metrics plot to: {metrics_path}")
    plt.close()
    
    # ============================================
    # 2. CONFUSION MATRIX (Absolute Counts)
    # ============================================
    print("📊 Creating confusion matrix (absolute counts)...")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix (Absolute Counts) - TFLite Model', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_abs_path = os.path.join(save_dir, "confusion_matrix_absolute.png")
    plt.savefig(cm_abs_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved absolute confusion matrix to: {cm_abs_path}")
    plt.close()
    
    # ============================================
    # 3. CONFUSION MATRIX (Normalized - Decimal)
    # ============================================
    print("📊 Creating confusion matrix (normalized - decimal)...")
    
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Value'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=1
    )
    plt.title('Confusion Matrix (Normalized - Decimal) - TFLite Model', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved normalized confusion matrix to: {cm_norm_path}")
    plt.close()
    
    # ============================================
    # 4. CONFUSION MATRIX (Normalized - Percent)
    # ============================================
    print("📊 Creating confusion matrix (normalized - percent)...")
    
    plt.figure(figsize=(12, 10))
    cm_percent = cm_normalized * 100  # Convert to percentage
    
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=100,
        annot_kws={'fontsize': 10, 'fontweight': 'bold'}
    )
    plt.title('Confusion Matrix (Normalized - Percent View) - TFLite Model', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_percent_path = os.path.join(save_dir, "confusion_matrix_percent.png")
    plt.savefig(cm_percent_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved percent confusion matrix to: {cm_percent_path}")
    plt.close()
    
    # ============================================
    # 5. CONFUSION MATRIX (Default - Percent, as confusion_matrix.png)
    # ============================================
    print("📊 Creating default confusion matrix (confusion_matrix.png)...")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=100,
        annot_kws={'fontsize': 10, 'fontweight': 'bold'}
    )
    plt.title('Confusion Matrix - TFLite Model', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_default_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_default_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved default confusion matrix to: {cm_default_path}")
    plt.close()
    
    print("✅ All visualizations created successfully!")

# -----------------------------
# 🚀 MAIN EVALUATION FUNCTION
# -----------------------------
def main(use_full_dataset=False, model_path=None):
    """
    Main evaluation pipeline for TFLite model.
    
    Args:
        use_full_dataset: If True, evaluate on full dataset. If False, use test split.
        model_path: Optional custom path to TFLite model file. If None, searches backend and assets.
    """
    print("=" * 80)
    print("TFLITE MODEL EVALUATION SCRIPT")
    print("=" * 80)
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Display model search locations and fallback strategy
    print("🔍 TFLite Model Search Strategy:")
    print(f"  1. Primary: {BACKEND_MODEL_DIR}")
    print(f"  2. Fallback: {ASSETS_MODEL_DIR} (if backend is missing)")
    print()
    
    # Check availability upfront
    backend_model_path = os.path.join(BACKEND_MODEL_DIR, TFLITE_MODEL_NAME)
    assets_model_path = os.path.join(ASSETS_MODEL_DIR, TFLITE_MODEL_NAME)
    backend_available = os.path.exists(backend_model_path)
    assets_available = os.path.exists(assets_model_path)
    
    if backend_available:
        print(f"✅ Backend TFLite model available: {backend_model_path}")
    else:
        print(f"⚠️  Backend TFLite model missing: {backend_model_path}")
    
    if assets_available:
        print(f"✅ Assets TFLite model available: {assets_model_path}")
        if not backend_available:
            print(f"   → Will use assets model as fallback")
    else:
        print(f"⚠️  Assets TFLite model missing: {assets_model_path}")
    
    print()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(RESULTS_DIR, f"evaluation_tflite_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Step 1: Load dataset
        images, labels = load_dataset()
        
        # Step 2: Prepare test data
        X_test, y_test_categorical, y_test_labels, label_encoder = prepare_test_data(
            images, labels, create_test_split=not use_full_dataset
        )
        
        # Step 3: Load TFLite model
        interpreter, input_details, output_details = load_tflite_model(custom_model_path=model_path)
        
        # Step 4: Run predictions
        y_pred, y_pred_proba = run_predictions_tflite(interpreter, input_details, output_details, X_test)
        
        # Step 5: Extract true labels
        y_true = extract_true_labels(y_test_categorical)
        
        # Step 6: Generate metrics
        report, cm, accuracy, precision, recall, f1, support = generate_metrics(
            y_true, y_pred, label_encoder, save_dir
        )
        
        # Step 7: Visualize results
        visualize_results(
            cm, 
            label_encoder.classes_, 
            precision, 
            recall, 
            f1, 
            save_dir
        )
        
        print("\n" + "=" * 80)
        print("✅ TFLITE EVALUATION COMPLETE")
        print("=" * 80)
        print(f"📁 Results saved to: {save_dir}")
        print(f"📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nGenerated files:")
        print(f"  📄 classification_report.json")
        print(f"  📄 detailed_metrics.txt (used by plot_metrics.py)")
        print(f"  📊 per_class_metrics.png")
        print(f"  📊 confusion_matrix.png (default - percent view)")
        print(f"  📊 confusion_matrix_absolute.png")
        print(f"  📊 confusion_matrix_normalized.png")
        print(f"  📊 confusion_matrix_percent.png")
        print(f"\n💡 Tip: Use plot_metrics.py in JupyterLab to visualize these results!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TFLite eye disease detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test split (default, auto-detect TFLite model)
  python evaluate_tflite_model.py
  
  # Evaluate on full dataset
  python evaluate_tflite_model.py --full-dataset
  
  # Use custom TFLite model path
  python evaluate_tflite_model.py --model-path /path/to/model.tflite
        """
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluate on full dataset instead of test split (default: False)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to TFLite model file (default: auto-detect from backend/assets)"
    )
    
    args = parser.parse_args()
    main(
        use_full_dataset=args.full_dataset,
        model_path=args.model_path
    )


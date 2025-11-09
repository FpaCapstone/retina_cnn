"""
Enhanced 5-Stage Pipeline for Eye Disease Detection
Improves accuracy for camera-captured images

Pipeline Stages:
1. Image Quality AI - Detects blur, glare, crop issues
2. Preprocessing Enhancer - Fixes contrast, sharpness, alignment
3. Normal-vs-Abnormal AI - Quickly filters healthy eyes
4. Disease Classifier - Classifies 5 conditions (existing Retina CNN)
5. Confidence/Consistency Validator - Checks confidence & consistency
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import json

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "backend", "models")
ASSETS_MODEL_DIR = os.path.join(PROJECT_ROOT, "assets", "images", "models")

MODEL_NAME = "outer_eye_mobilenetv2.h5"
IMG_SIZE = (224, 224)
DISEASES = ["Normal", "Uveitis", "Conjunctivitis", "Cataract", "Eyelid Drooping"]

# Quality thresholds
BLUR_THRESHOLD = 100.0  # Laplacian variance threshold
GLARE_THRESHOLD = 0.3  # Brightness threshold
MIN_CROP_RATIO = 0.7  # Minimum eye coverage ratio

# ============================================
# STAGE 1: IMAGE QUALITY AI
# ============================================

def detect_blur(image: np.ndarray):
    """
    Detect blur using Laplacian variance.
    
    Returns:
        (blur_score, is_blurry)
        - blur_score: Laplacian variance (higher = sharper)
        - is_blurry: True if image is too blurry
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < BLUR_THRESHOLD
    return float(laplacian_var), is_blurry

def detect_glare(image: np.ndarray):
    """
    Detect glare/overexposure.
    
    Returns:
        (glare_score, has_glare)
        - glare_score: Average brightness (0-1)
        - has_glare: True if image has excessive glare
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(gray) / 255.0
    has_glare = brightness > GLARE_THRESHOLD
    return brightness, has_glare

def detect_crop_issues(image: np.ndarray):
    """
    Detect if image is properly cropped (eye is centered and visible).
    
    Returns:
        (coverage_ratio, has_crop_issue)
        - coverage_ratio: Ratio of non-black pixels (0-1)
        - has_crop_issue: True if eye is not properly visible
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    # Detect edges to find eye region
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate coverage (non-zero pixels)
    total_pixels = gray.size
    non_zero_pixels = np.count_nonzero(gray > 10)  # Threshold for "visible" pixels
    coverage_ratio = non_zero_pixels / total_pixels
    
    has_crop_issue = coverage_ratio < MIN_CROP_RATIO
    
    return coverage_ratio, has_crop_issue

def load_quality_classifier():
    """Load ML-based quality classifier if available."""
    quality_model_path = os.path.join(MODEL_DIR, "quality_classifier.h5")
    quality_assets_path = os.path.join(ASSETS_MODEL_DIR, "quality_classifier.h5")
    
    if os.path.exists(quality_model_path):
        try:
            import sys
            print("🤖 Loading ML quality classifier...", file=sys.stderr)
            model = load_model(quality_model_path)
            
            # Load label encoder
            import json
            encoder_path = os.path.join(MODEL_DIR, "quality_label_encoder.json")
            if os.path.exists(encoder_path):
                with open(encoder_path, 'r') as f:
                    encoder_data = json.load(f)
                return model, encoder_data
            return model, None
        except Exception as e:
            import sys
            print(f"⚠️  Could not load ML quality classifier: {e}", file=sys.stderr)
            return None, None
    elif os.path.exists(quality_assets_path):
        try:
            import sys
            print("🤖 Loading ML quality classifier from assets...", file=sys.stderr)
            model = load_model(quality_assets_path)
            return model, None
        except Exception as e:
            import sys
            print(f"⚠️  Could not load ML quality classifier: {e}", file=sys.stderr)
            return None, None
    
    return None, None

def assess_image_quality_ml(image_path: str, model, encoder_data=None) -> dict:
    """
    ML-based image quality assessment.
    
    Returns:
        Dictionary with quality metrics and recommendations
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image_rgb, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batched = np.expand_dims(img_normalized, axis=0)
    
    # Get ML prediction
    predictions = model.predict(img_batched, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    
    # Map prediction to quality class
    quality_classes = ["good", "blurry", "glare", "poor_crop", "multiple_issues"]
    if encoder_data:
        quality_classes = encoder_data.get('classes', quality_classes)
    
    predicted_class = quality_classes[predicted_idx]
    
    # Convert ML prediction to traditional format
    is_blurry = predicted_class in ["blurry", "multiple_issues"]
    has_glare = predicted_class in ["glare", "multiple_issues"]
    has_crop_issue = predicted_class in ["poor_crop", "multiple_issues"]
    
    # Calculate quality score from ML confidence
    if predicted_class == "good":
        quality_score = confidence
    elif predicted_class == "multiple_issues":
        quality_score = 0.3
    else:
        quality_score = 0.5  # Single issue
    
    issues = []
    if is_blurry:
        issues.append("blur")
    if has_glare:
        issues.append("glare")
    if has_crop_issue:
        issues.append("crop")
    
    return {
        'quality_score': float(quality_score),
        'blur_score': None,  # ML doesn't provide individual scores
        'is_blurry': is_blurry,
        'glare_score': None,
        'has_glare': has_glare,
        'coverage_ratio': None,
        'has_crop_issue': has_crop_issue,
        'issues': issues,
        'needs_preprocessing': len(issues) > 0,
        'recommendation': 'retake' if quality_score < 0.5 else 'proceed',
        'method': 'ml',
        'ml_confidence': confidence,
        'ml_predicted_class': predicted_class
    }

def assess_image_quality(image_path: str, use_ml: bool = True) -> dict:
    """
    Stage 1: Comprehensive image quality assessment.
    
    Uses ML-based classifier if available, otherwise falls back to traditional CV methods.
    
    Args:
        image_path: Path to image
        use_ml: Whether to try ML classifier first (default: True)
    
    Returns:
        Dictionary with quality metrics and recommendations
    """
    # Try ML-based assessment first if requested and available
    if use_ml:
        quality_model, encoder_data = load_quality_classifier()
        if quality_model is not None:
            try:
                return assess_image_quality_ml(image_path, quality_model, encoder_data)
            except Exception as e:
                import sys
                print(f"⚠️  ML quality assessment failed, using CV fallback: {e}", file=sys.stderr)
    
    # Fallback to traditional CV methods
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    blur_score, is_blurry = detect_blur(image_rgb)
    glare_score, has_glare = detect_glare(image_rgb)
    coverage_ratio, has_crop_issue = detect_crop_issues(image_rgb)
    
    quality_score = (
        (1.0 if not is_blurry else 0.3) * 0.4 +
        (1.0 if not has_glare else 0.5) * 0.3 +
        coverage_ratio * 0.3
    )
    
    issues = []
    if is_blurry:
        issues.append("blur")
    if has_glare:
        issues.append("glare")
    if has_crop_issue:
        issues.append("crop")
    
    return {
        'quality_score': float(quality_score),
        'blur_score': blur_score,
        'is_blurry': is_blurry,
        'glare_score': glare_score,
        'has_glare': has_glare,
        'coverage_ratio': coverage_ratio,
        'has_crop_issue': has_crop_issue,
        'issues': issues,
        'needs_preprocessing': len(issues) > 0,
        'recommendation': 'retake' if quality_score < 0.5 else 'proceed',
        'method': 'cv'  # Traditional computer vision
    }

# ============================================
# STAGE 2: PREPROCESSING ENHANCER
# ============================================

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
    return enhanced

def enhance_sharpness(image: np.ndarray) -> np.ndarray:
    """Enhance sharpness using unsharp masking."""
    # Convert to float
    image_float = image.astype(np.float32) / 255.0
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image_float, (0, 0), 2.0)
    
    # Unsharp mask
    sharpened = cv2.addWeighted(image_float, 1.5, blurred, -0.5, 0)
    
    # Clip and convert back
    sharpened = np.clip(sharpened, 0, 1) * 255.0
    return sharpened.astype(np.uint8)

def reduce_glare(image: np.ndarray) -> np.ndarray:
    """Reduce glare/overexposure."""
    # Convert to LAB color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gamma correction to L channel
        l_float = l.astype(np.float32) / 255.0
        gamma = 1.2  # Darken slightly
        l_corrected = np.power(l_float, gamma) * 255.0
        l_corrected = l_corrected.astype(np.uint8)
        
        # Merge back
        enhanced = cv2.merge([l_corrected, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    else:
        # For grayscale
        l_float = image.astype(np.float32) / 255.0
        gamma = 1.2
        enhanced = (np.power(l_float, gamma) * 255.0).astype(np.uint8)
    
    return enhanced

def preprocess_image(image_path: str, quality_report: dict):
    """
    Stage 2: Enhance image based on quality issues detected.
    
    Returns:
        (enhanced_image, preprocessing_log)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced = image_rgb.copy()
    preprocessing_steps = []
    
    # Apply enhancements based on detected issues
    if quality_report['is_blurry']:
        enhanced = enhance_sharpness(enhanced)
        preprocessing_steps.append('sharpness')
    
    if quality_report['has_glare']:
        enhanced = reduce_glare(enhanced)
        preprocessing_steps.append('glare_reduction')
    
    # Always enhance contrast slightly for better feature detection
    enhanced = enhance_contrast(enhanced)
    preprocessing_steps.append('contrast')
    
    return enhanced, {
        'preprocessing_applied': preprocessing_steps,
        'original_shape': image_rgb.shape,
        'enhanced_shape': enhanced.shape
    }

# ============================================
# STAGE 3: NORMAL-VS-ABNORMAL AI
# ============================================

def create_normal_vs_abnormal_classifier(model):
    """
    Create a binary normal/abnormal classifier from the existing multi-class model.
    Uses the model's 'Normal' class probability as the normal score.
    """
    def classify_normal_abnormal(image_array: np.ndarray) -> dict:
        # Preprocess for model
        img_resized = cv2.resize(image_array, IMG_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batched = np.expand_dims(img_normalized, axis=0)
        
        # Get predictions
        predictions = model.predict(img_batched, verbose=0)
        
        # Find Normal class index
        normal_idx = DISEASES.index("Normal")
        normal_prob = float(predictions[0][normal_idx])
        abnormal_prob = 1.0 - normal_prob
        
        is_normal = normal_prob > 0.7  # Threshold for "normal"
        
        return {
            'is_normal': is_normal,
            'normal_probability': normal_prob,
            'abnormal_probability': abnormal_prob,
            'confidence': max(normal_prob, abnormal_prob)
        }
    
    return classify_normal_abnormal

def filter_normal_eyes(image_array: np.ndarray, model, normal_classifier):
    """
    Stage 3: Quick normal/abnormal filter.
    
    Returns:
        Dictionary with normal/abnormal classification
    """
    result = normal_classifier(image_array)
    
    if result['is_normal']:
        return {
            'stage': 'normal_filter',
            'result': 'normal',
            'confidence': result['normal_probability'],
            'skip_disease_classification': True,  # Skip to stage 5
            'message': 'Eye appears normal. No disease classification needed.'
        }
    else:
        return {
            'stage': 'normal_filter',
            'result': 'abnormal',
            'confidence': result['abnormal_probability'],
            'skip_disease_classification': False,  # Proceed to stage 4
            'message': 'Abnormalities detected. Proceeding with disease classification.'
        }

# ============================================
# STAGE 4: DISEASE CLASSIFIER (Existing Model)
# ============================================

def classify_diseases(image_array: np.ndarray, model):
    """
    Stage 4: Multi-class disease classification using existing Retina CNN.
    
    Returns:
        Dictionary with disease predictions and probabilities
    """
    # Preprocess for model
    img_resized = cv2.resize(image_array, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batched = np.expand_dims(img_normalized, axis=0)
    
    # Get predictions
    predictions = model.predict(img_batched, verbose=0)[0]
    
    # Get probabilities for each disease
    disease_probs = {}
    for i, disease in enumerate(DISEASES):
        disease_probs[disease] = float(predictions[i])
    
    # Get top prediction
    predicted_idx = np.argmax(predictions)
    predicted_disease = DISEASES[predicted_idx]
    confidence = float(predictions[predicted_idx])
    
    return {
        'predicted_disease': predicted_disease,
        'confidence': confidence,
        'probabilities': disease_probs,
        'all_predictions': {d: float(p) for d, p in zip(DISEASES, predictions)}
    }

# ============================================
# STAGE 5: CONFIDENCE/CONSISTENCY VALIDATOR
# ============================================

def validate_confidence(prediction: dict, quality_report: dict):
    """
    Stage 5: Validate prediction confidence and consistency.
    
    Returns:
        Dictionary with validation results and final recommendation
    """
    confidence = prediction['confidence']
    quality_score = quality_report['quality_score']
    
    # Adjust confidence based on image quality
    quality_adjusted_confidence = confidence * quality_score
    
    # Consistency check: if top 2 predictions are close, lower confidence
    probs = list(prediction['probabilities'].values())
    probs_sorted = sorted(probs, reverse=True)
    top2_diff = probs_sorted[0] - probs_sorted[1] if len(probs_sorted) > 1 else 1.0
    
    consistency_score = min(1.0, top2_diff * 2)  # Higher diff = more consistent
    final_confidence = quality_adjusted_confidence * consistency_score
    
    # Determine if prediction is reliable
    is_reliable = (
        final_confidence > 0.6 and
        quality_score > 0.5 and
        consistency_score > 0.5
    )
    
    recommendation = 'accept' if is_reliable else 'retake'
    
    if not is_reliable:
        reasons = []
        if final_confidence <= 0.6:
            reasons.append("low_confidence")
        if quality_score <= 0.5:
            reasons.append("poor_image_quality")
        if consistency_score <= 0.5:
            reasons.append("inconsistent_predictions")
    else:
        reasons = []
    
    return {
        'original_confidence': confidence,
        'quality_adjusted_confidence': quality_adjusted_confidence,
        'final_confidence': final_confidence,
        'consistency_score': consistency_score,
        'is_reliable': is_reliable,
        'recommendation': recommendation,
        'reasons': reasons,
        'quality_score': quality_score
    }

# ============================================
# MAIN PIPELINE
# ============================================

def load_model_with_fallback():
    """Load model from backend or assets with fallback."""
    import sys
    backend_path = os.path.join(MODEL_DIR, MODEL_NAME)
    assets_path = os.path.join(ASSETS_MODEL_DIR, MODEL_NAME)
    
    if os.path.exists(backend_path):
        print(f"✅ Loading model from backend: {backend_path}", file=sys.stderr)
        return load_model(backend_path)
    elif os.path.exists(assets_path):
        print(f"✅ Loading model from assets: {assets_path}", file=sys.stderr)
        return load_model(assets_path)
    else:
        raise FileNotFoundError(f"Model not found in {backend_path} or {assets_path}")

def run_enhanced_pipeline(image_path: str, enable_stages: dict = None):
    """
    Run the complete 5-stage enhanced pipeline.
    
    Args:
        image_path: Path to input image
        enable_stages: Dict to enable/disable stages (default: all enabled)
            Example: {'quality_check': True, 'preprocessing': True, ...}
    
    Returns:
        Complete pipeline results with all stage outputs
    """
    if enable_stages is None:
        enable_stages = {
            'quality_check': True,
            'preprocessing': True,
            'normal_filter': True,
            'disease_classification': True,
            'validation': True
        }
    
    results = {
        'stages': {},
        'final_prediction': None,
        'recommendation': None
    }
    
    # Load model once
    model = load_model_with_fallback()
    normal_classifier = create_normal_vs_abnormal_classifier(model)
    
    # Stage 1: Image Quality AI
    if enable_stages.get('quality_check', True):
        import sys
        print("🔍 Stage 1: Assessing image quality...", file=sys.stderr)
        # Try ML first, fallback to CV
        use_ml = enable_stages.get('use_ml_quality', True)
        quality_report = assess_image_quality(image_path, use_ml=use_ml)
        results['stages']['quality'] = quality_report
        
        if quality_report['recommendation'] == 'retake':
            results['recommendation'] = 'retake'
            results['reason'] = f"Image quality too low: {quality_report['issues']}"
            results['final_prediction'] = None
            results['final_confidence'] = 0.0
            # Continue to output JSON (don't return early)
    else:
        quality_report = {'quality_score': 1.0, 'needs_preprocessing': False}
    
    # Stage 2: Preprocessing Enhancer
    if enable_stages.get('preprocessing', True) and quality_report.get('needs_preprocessing', False):
        import sys
        print("✨ Stage 2: Enhancing image...", file=sys.stderr)
        enhanced_image, preprocessing_log = preprocess_image(image_path, quality_report)
        results['stages']['preprocessing'] = preprocessing_log
        image_for_analysis = enhanced_image
    else:
        # Load original image
        image_for_analysis = cv2.imread(image_path)
        image_for_analysis = cv2.cvtColor(image_for_analysis, cv2.COLOR_BGR2RGB)
        results['stages']['preprocessing'] = {'preprocessing_applied': []}
    
    # Stage 3: Normal vs Abnormal Filter
    if enable_stages.get('normal_filter', True):
        import sys
        print("🔬 Stage 3: Checking normal vs abnormal...", file=sys.stderr)
        normal_result = filter_normal_eyes(image_for_analysis, model, normal_classifier)
        results['stages']['normal_filter'] = normal_result
        
        if normal_result.get('skip_disease_classification', False):
            # Skip to validation
            results['final_prediction'] = {
                'disease': 'Normal',
                'confidence': normal_result['confidence'],
                'probabilities': {'Normal': normal_result['confidence']}
            }
            results['stages']['disease_classification'] = {'skipped': True, 'reason': 'normal_eye_detected'}
    else:
        normal_result = {'skip_disease_classification': False}
    
    # Stage 4: Disease Classifier
    if enable_stages.get('disease_classification', True) and not normal_result.get('skip_disease_classification', False):
        import sys
        print("🏥 Stage 4: Classifying diseases...", file=sys.stderr)
        disease_prediction = classify_diseases(image_for_analysis, model)
        results['stages']['disease_classification'] = disease_prediction
        results['final_prediction'] = disease_prediction
    elif not normal_result.get('skip_disease_classification', False):
        # If disease classification is disabled, still run it
        disease_prediction = classify_diseases(image_for_analysis, model)
        results['stages']['disease_classification'] = disease_prediction
        results['final_prediction'] = disease_prediction
    
    # Stage 5: Confidence/Consistency Validator
    if enable_stages.get('validation', True) and results['final_prediction']:
        import sys
        print("✅ Stage 5: Validating confidence and consistency...", file=sys.stderr)
        validation_result = validate_confidence(results['final_prediction'], quality_report)
        results['stages']['validation'] = validation_result
        
        # Update final recommendation
        if validation_result['recommendation'] == 'retake':
            results['recommendation'] = 'retake'
            results['reason'] = validation_result['reasons']
        else:
            results['recommendation'] = 'accept'
            results['final_confidence'] = validation_result['final_confidence']
    else:
        results['recommendation'] = 'accept'
    
    return results

# ============================================
# CLI / MAIN
# ============================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enhanced_pipeline.py <image_path> [--disable-stage STAGE]")
        print("Stages: quality_check, preprocessing, normal_filter, disease_classification, validation")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Parse disable flags
    enable_stages = {
        'quality_check': True,
        'preprocessing': True,
        'normal_filter': True,
        'disease_classification': True,
        'validation': True
    }
    
    if '--disable-stage' in sys.argv:
        idx = sys.argv.index('--disable-stage')
        if idx + 1 < len(sys.argv):
            stage = sys.argv[idx + 1]
            if stage in enable_stages:
                enable_stages[stage] = False
                print(f"⚠️  Disabled stage: {stage}")
    
    try:
        results = run_enhanced_pipeline(image_path, enable_stages)
        
        # Output JSON for backend consumption (stdout only, no print statements)
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        results_serializable = convert_to_json_serializable(results)
        
        # Output JSON to stdout (for backend to parse)
        print(json.dumps(results_serializable, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


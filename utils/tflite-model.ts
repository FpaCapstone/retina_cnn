import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';
import { Asset } from 'expo-asset';
import { DiseaseType } from '@/types/disease';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

// Use TensorFlow.js for model loading (compatible with Expo managed workflow)
// For TFLite models, they need to be converted to TensorFlow.js format

let tfModel: tf.GraphModel | tf.LayersModel | null = null;
let isModelLoaded = false;
let modelPath: string | null = null;

const DISEASES: DiseaseType[] = ['normal', 'uveitis', 'conjunctivitis', 'cataract', 'eyelid_drooping'];
const DISEASE_NAMES = ['Normal', 'Uveitis', 'Conjunctivitis', 'Cataract', 'Eyelid Drooping'];

/**
 * Initialize TensorFlow.js
 */
async function initializeTensorFlow(): Promise<boolean> {
  try {
    console.log('[TF.js] Initializing TensorFlow.js...');
    await tf.ready();
    console.log('[TF.js] TensorFlow.js backend:', tf.getBackend());
    console.log('[TF.js] TensorFlow.js initialized successfully');
    return true;
  } catch (error) {
    console.error('[TF.js] Failed to initialize TensorFlow:', error);
    return false;
  }
}

/**
 * Load TensorFlow.js model from assets
 * Note: TFLite models need to be converted to TensorFlow.js format (model.json + weights)
 * For now, we'll try to load from a TensorFlow.js model if available
 */
export async function loadTFLiteModel(): Promise<boolean> {
  try {
    if (isModelLoaded && tfModel) {
      console.log('[TF.js] Model already loaded');
      return true;
    }

    // Initialize TensorFlow.js
    const initialized = await initializeTensorFlow();
    if (!initialized) {
      console.warn('[TF.js] TensorFlow.js not initialized');
      return false;
    }

    // Try to load TensorFlow.js model from assets
    // First, check if we have a model.json file in assets
    try {
      // For TensorFlow.js, we need model.json and weight files
      // Since we only have .tflite, we'll need to convert it or use a different approach
      // For now, we'll check if there's a model.json in the assets/models directory
      
      // Note: TFLite models cannot be directly loaded with TensorFlow.js
      // They need to be converted to TensorFlow.js format first
      // This function will return false, and the app will fall back to backend or offline analysis
      
      console.log('[TF.js] TFLite model found, but TensorFlow.js requires converted format');
      console.log('[TF.js] To use TensorFlow.js, convert .tflite to model.json + weights');
      console.log('[TF.js] Falling back to backend or offline analysis');
      
      return false;
    } catch (error) {
      console.warn('[TF.js] Model not available in TensorFlow.js format:', error);
      return false;
    }
  } catch (error) {
    console.error('[TF.js] Error loading model:', error);
    return false;
  }
}

/**
 * Preprocess image for TensorFlow.js inference
 * Properly decodes, resizes, and normalizes image for MobileNetV2 input (224x224x3)
 */
async function preprocessImageForTF(imageUri: string): Promise<tf.Tensor3D> {
  try {
    console.log('[TF.js] Preprocessing image:', imageUri);
    
    // Ensure TensorFlow.js is ready
    await tf.ready();
    
    // Read image as base64
    let base64: string;
    let imageFormat = 'jpeg';
    
    if (imageUri.startsWith('data:image')) {
      // Extract base64 and format from data URI
      const match = imageUri.match(/data:image\/(\w+);base64,(.+)/);
      if (match) {
        imageFormat = match[1] || 'jpeg';
        base64 = match[2];
      } else {
        // Fallback: extract base64 after comma
        base64 = imageUri.split(',')[1];
      }
    } else {
      // Read from file system
      base64 = await FileSystem.readAsStringAsync(imageUri, {
        encoding: 'base64' as const,
      });
      
      // Detect format from file extension
      if (imageUri.toLowerCase().endsWith('.png')) {
        imageFormat = 'png';
      } else if (imageUri.toLowerCase().endsWith('.jpg') || imageUri.toLowerCase().endsWith('.jpeg')) {
        imageFormat = 'jpeg';
      }
    }
    
    // Create data URI for image loading
    const imageData = `data:image/${imageFormat};base64,${base64}`;
    
    // Decode image to tensor using TensorFlow.js
    const imageTensor = await new Promise<tf.Tensor3D>((resolve, reject) => {
      const img = new Image();
      
      img.onload = () => {
        try {
          // Convert image to tensor (shape: [height, width, 3])
          const tensor = tf.browser.fromPixels(img);
          
          // Resize to 224x224 (MobileNetV2 input size)
          const resized = tf.image.resizeBilinear(tensor, [224, 224]) as tf.Tensor3D;
          
          // Normalize pixel values from [0, 255] to [0, 1]
          const normalized = resized.div(255.0) as tf.Tensor3D;
          
          // Clean up intermediate tensors
          tensor.dispose();
          resized.dispose();
          
          resolve(normalized);
        } catch (err) {
          reject(err);
        }
      };
      
      img.onerror = (error) => {
        reject(new Error(`Failed to load image: ${error}`));
      };
      
      img.src = imageData;
    });
    
    console.log('[TF.js] Image preprocessed successfully. Shape:', imageTensor.shape);
    
    return imageTensor;
  } catch (error) {
    console.error('[TF.js] Error preprocessing image:', error);
    throw new Error(`Image preprocessing failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Load TensorFlow.js model from assets (model.json format)
 * This function loads a TensorFlow.js model if available
 */
async function loadTFModelFromAssets(): Promise<tf.GraphModel | tf.LayersModel | null> {
  try {
    // Initialize TensorFlow.js
    await tf.ready();
    
    console.log('[TF.js] Attempting to load TensorFlow.js model from assets...');
    
    // Try to load model.json from assets
    // First, check if model.json exists in assets/images/models/
    // NOTE: This project currently ships only a .tflite model. To use TensorFlow.js
    // inference on-device, you must convert the .tflite model to TensorFlow.js format
    // (model.json + weights) using the tensorflowjs_converter CLI. Because the converted
    // files are not present, we return null here which triggers the caller to fall back to
    // backend or offline analysis.
    console.log('[TF.js] TensorFlow.js model files not bundled. Provide model.json + weights to enable on-device TF.js inference.');
    console.log('[TF.js] Conversion command:');
    console.log('tensorflowjs_converter --input_format=tf_lite --output_format=tfjs_graph_model outer_eye_mobilenetv2.tflite assets/images/models/outer_eye_mobilenetv2/');
    return null;
  } catch (error) {
    console.warn('[TF.js] Could not load model from assets:', error);
    return null;
  }
}

/**
 * Run inference using TensorFlow.js model
 * Falls back gracefully if model is not available
 */
export async function predictWithTFLite(imageUri: string): Promise<{
  primaryDisease: DiseaseType;
  detections: Array<{
    disease: DiseaseType;
    confidence: number;
    percentage: number;
  }>;
}> {
  try {
    // Try to load TensorFlow.js model
    if (!tfModel) {
      tfModel = await loadTFModelFromAssets();
    }

    if (!tfModel) {
      // Model not available - this is expected if .tflite hasn't been converted
      throw new Error('TensorFlow.js model not available. TFLite model needs to be converted to TensorFlow.js format.');
    }

    // Preprocess image
    const preprocessedImage = await preprocessImageForTF(imageUri);
    
    // Add batch dimension [1, 224, 224, 3]
    const batchedImage = preprocessedImage.expandDims(0);
    
    // Run inference
    console.log('[TF.js] Running inference...');
    const predictions = tfModel.predict(batchedImage) as tf.Tensor;
    const probabilitiesArray = await predictions.data();
    
    // Clean up tensors
    preprocessedImage.dispose();
    batchedImage.dispose();
    predictions.dispose();

    // Parse output (assuming output is array of probabilities for each disease)
    const probabilities = Array.from(probabilitiesArray);
    
    // Ensure we have probabilities for all diseases
    const paddedProbabilities = [...probabilities];
    while (paddedProbabilities.length < DISEASES.length) {
      paddedProbabilities.push(0);
    }
    
    // Map to disease types
    const detections = paddedProbabilities.slice(0, DISEASES.length).map((prob: number, index: number) => {
      const disease = DISEASES[index] || 'normal';
      return {
        disease,
        confidence: Math.max(0, Math.min(1, prob)), // Clamp to [0, 1]
        percentage: Math.max(0, Math.min(100, prob * 100)),
      };
    });

    // Sort by confidence
    detections.sort((a, b) => b.confidence - a.confidence);

    const primaryDisease = detections[0]?.disease || 'normal';

    console.log('[TF.js] Prediction complete:', { primaryDisease, detections });

    return {
      primaryDisease,
      detections: [detections[0]], // Only highest prediction
    };
  } catch (error) {
    console.error('[TF.js] Prediction error:', error);
    throw error;
  }
}

/**
 * Check if TensorFlow.js model is available
 */
export async function checkTFLiteModelAvailable(): Promise<boolean> {
  try {
    if (isModelLoaded && tfModel) {
      return true;
    }

    // Try to load model
    const loaded = await loadTFLiteModel();
    if (loaded && tfModel) {
      return true;
    }

    // Try to load from assets
    tfModel = await loadTFModelFromAssets();
    if (tfModel) {
      isModelLoaded = true;
      return true;
    }

    return false;
  } catch (error) {
    console.error('[TF.js] Error checking model availability:', error);
    return false;
  }
}

/**
 * Dispose TensorFlow.js model
 */
export function disposeTFLiteModel(): void {
  if (tfModel) {
    try {
      tfModel.dispose();
      tfModel = null;
      isModelLoaded = false;
      modelPath = null;
      console.log('[TF.js] Model disposed');
    } catch (error) {
      console.error('[TF.js] Error disposing model:', error);
    }
  }
}


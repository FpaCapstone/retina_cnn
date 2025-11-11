import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';

/**
 * Convert image URI to base64 data URI for backend upload
 */
export async function convertImageToBase64(imageUri: string): Promise<string> {
  try {
    // If already base64, return as is
    if (imageUri.startsWith('data:image')) {
      return imageUri;
    }

    // If it's a file URI, read and convert
    if (imageUri.startsWith('file://') || imageUri.startsWith('/')) {
      const base64 = await FileSystem.readAsStringAsync(imageUri, {
        encoding: 'base64' as const,
      });
      
      // Determine MIME type from file extension
      let mimeType = 'image/jpeg';
      if (imageUri.toLowerCase().endsWith('.png')) {
        mimeType = 'image/png';
      } else if (imageUri.toLowerCase().endsWith('.webp')) {
        mimeType = 'image/webp';
      }
      
      return `data:${mimeType};base64,${base64}`;
    }

    // For web, try to fetch and convert
    if (Platform.OS === 'web' && imageUri.startsWith('http')) {
      const response = await fetch(imageUri);
      const blob = await response.blob();
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }

    // Return as is if we can't convert
    console.warn('[ImageUtils] Could not convert image URI to base64:', imageUri);
    return imageUri;
  } catch (error) {
    console.error('[ImageUtils] Error converting image to base64:', error);
    return imageUri; // Return original on error
  }
}

/**
 * Check if image URI is already in base64 format
 */
export function isBase64Image(uri: string): boolean {
  return uri.startsWith('data:image');
}


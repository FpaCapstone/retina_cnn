/**
 * Model Evaluation Metrics
 * Loads and provides access to model evaluation metrics
 */

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ModelMetrics {
  overallAccuracy: number;
  classes: Record<string, ClassMetrics>;
  macroAvg: {
    precision: number;
    recall: number;
    f1Score: number;
  };
  weightedAvg: {
    precision: number;
    recall: number;
    f1Score: number;
  };
  testSetSize: number;
  evaluationDate?: string;
}

/**
 * Get the latest model evaluation metrics
 * These are static values based on the most recent evaluation
 */
export function getModelMetrics(): ModelMetrics {
  // Based on evaluation_20251109_164846/detailed_metrics.txt
  return {
    overallAccuracy: 0.9913, // 99.13%
    testSetSize: 2298,
    evaluationDate: '2025-11-09',
    classes: {
      'Normal': {
        precision: 0.9969,
        recall: 0.9985,
        f1Score: 0.9977,
        support: 649,
      },
      'Cataract': {
        precision: 0.9926,
        recall: 0.9926,
        f1Score: 0.9926,
        support: 544,
      },
      'Eyelid Drooping': {
        precision: 0.9923,
        recall: 0.9829,
        f1Score: 0.9876,
        support: 525,
      },
      'Conjunctivitis': {
        precision: 0.9888,
        recall: 0.9888,
        f1Score: 0.9888,
        support: 357,
      },
      'Uveitis': {
        precision: 0.9736,
        recall: 0.9910,
        f1Score: 0.9822,
        support: 223,
      },
    },
    macroAvg: {
      precision: 0.9888,
      recall: 0.9908,
      f1Score: 0.9898,
    },
    weightedAvg: {
      precision: 0.9913,
      recall: 0.9913,
      f1Score: 0.9913,
    },
  };
}

/**
 * Format percentage for display
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Get metrics summary text
 */
export function getMetricsSummary(): string {
  const metrics = getModelMetrics();
  return `Overall Accuracy: ${formatPercentage(metrics.overallAccuracy)} | Test Set: ${metrics.testSetSize} images`;
}


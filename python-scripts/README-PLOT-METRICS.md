# Plot Metrics - JupyterLab Usage Guide

## Overview

`plot_metrics.py` is a standalone Python script for visualizing model evaluation metrics. It's designed to work seamlessly in JupyterLab.

## Quick Start in JupyterLab

### Method 1: Load from Evaluation Results (Easiest)

If you've already run `evaluate_model.py`, simply load and plot:

```python
from plot_metrics import plot_from_evaluation_results

# Load and plot from evaluation results directory
results_dir = "evaluation_results/evaluation_20240101_120000"  # Update with your directory

plot_from_evaluation_results(results_dir, show_plots=True)
```

This will automatically:
- Load all metrics from the evaluation results
- Create all 4 visualizations
- Display them in JupyterLab
- Save them to the results directory

### Method 2: Plot Individual Metrics

For more control, plot specific metrics:

```python
from plot_metrics import (
    plot_per_class_metrics,
    plot_confusion_matrix_absolute,
    plot_confusion_matrix_normalized
)
import numpy as np

# Your metrics
class_names = ['Normal', 'Uveitis', 'Conjunctivitis', 'Cataract', 'Eyelid Drooping']
precision = np.array([0.90, 0.82, 0.85, 0.88, 0.84])
recall = np.array([0.91, 0.78, 0.88, 0.85, 0.82])
f1 = np.array([0.90, 0.80, 0.86, 0.86, 0.83])

# Plot per-class metrics
plot_per_class_metrics(class_names, precision, recall, f1)
```

```python
# Confusion matrix
cm = np.array([
    [85, 3, 2, 1, 1],   # Normal
    [4, 72, 5, 3, 8],   # Uveitis
    [2, 4, 81, 3, 2],   # Conjunctivitis
    [1, 2, 3, 81, 5],   # Cataract
    [2, 6, 2, 4, 78]    # Eyelid Drooping
])

# Plot confusion matrix - absolute counts
plot_confusion_matrix_absolute(cm, class_names)

# Plot confusion matrix - normalized percent
plot_confusion_matrix_normalized(cm, class_names, format_type='percent')
```

### Method 3: Plot All Metrics at Once

```python
from plot_metrics import plot_metrics
import numpy as np

# Your metrics
cm = np.array([
    [85, 3, 2, 1, 1],
    [4, 72, 5, 3, 8],
    [2, 4, 81, 3, 2],
    [1, 2, 3, 81, 5],
    [2, 6, 2, 4, 78]
])

class_names = ['Normal', 'Uveitis', 'Conjunctivitis', 'Cataract', 'Eyelid Drooping']
precision = np.array([0.90, 0.82, 0.85, 0.88, 0.84])
recall = np.array([0.91, 0.78, 0.88, 0.85, 0.82])
f1 = np.array([0.90, 0.80, 0.86, 0.86, 0.83])

# Plot all metrics
plot_metrics(cm, class_names, precision, recall, f1, save_dir='plots', show_plots=True)
```

## Available Functions

### Main Functions

1. **`plot_from_evaluation_results(results_dir, save_dir=None, show_plots=True)`**
   - Loads metrics from evaluation results directory
   - Creates all visualizations
   - Returns dictionary with saved file paths

2. **`plot_metrics(cm, class_names, precision, recall, f1, save_dir='plots', show_plots=True)`**
   - Main function to plot all metrics
   - Requires all metrics as input

3. **`load_evaluation_results(results_dir)`**
   - Loads metrics from saved evaluation results
   - Returns dictionary with all metrics

### Individual Plot Functions

1. **`plot_per_class_metrics(class_names, precision, recall, f1, save_path=None, show_plot=True)`**
   - Bar chart comparing Precision, Recall, F1-Score per class

2. **`plot_confusion_matrix_absolute(cm, class_names, save_path=None, show_plot=True)`**
   - Confusion matrix with absolute counts

3. **`plot_confusion_matrix_normalized(cm, class_names, format_type='decimal', save_path=None, show_plot=True)`**
   - Normalized confusion matrix
   - `format_type`: 'decimal' (0-1) or 'percent' (0-100%)

4. **`plot_confusion_matrices(cm, class_names, save_dir='plots', show_plots=True)`**
   - **NEW!** Convenience function to plot all 3 confusion matrix types at once
   - Creates: absolute, normalized decimal, and normalized percent views
   - Returns dictionary with paths to saved plots

## JupyterLab Tips

### Display Plots Inline

By default, plots are displayed inline in JupyterLab. To save without showing:

```python
plot_from_evaluation_results(results_dir, show_plots=False)
```

### Custom Save Directory

```python
plot_from_evaluation_results(
    results_dir="evaluation_results/evaluation_20240101_120000",
    save_dir="my_custom_plots",
    show_plots=True
)
```

### Load and Inspect Metrics

```python
from plot_metrics import load_evaluation_results

results = load_evaluation_results("evaluation_results/evaluation_20240101_120000")

# Check what's available
print(f"Confusion matrix available: {results['cm'] is not None}")
print(f"Class names: {results['class_names']}")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
print(f"F1-Score: {results['f1']}")
```

### Plot Only What You Need

```python
# Only plot per-class metrics (no confusion matrix needed)
from plot_metrics import plot_per_class_metrics

results = load_evaluation_results("evaluation_results/evaluation_20240101_120000")
plot_per_class_metrics(
    results['class_names'],
    results['precision'],
    results['recall'],
    results['f1']
)
```

```python
# Only plot confusion matrices
from plot_metrics import plot_confusion_matrices

results = load_evaluation_results("evaluation_results/evaluation_20240101_120000")
if results['cm'] is not None:
    plot_confusion_matrices(
        results['cm'],
        results['class_names'],
        save_dir='confusion_matrices'
    )
```

## Example JupyterLab Workflow

```python
# Cell 1: Import
from plot_metrics import plot_from_evaluation_results, load_evaluation_results
import numpy as np

# Cell 2: Load results
results_dir = "evaluation_results/evaluation_20240101_120000"
results = load_evaluation_results(results_dir)

# Cell 3: Inspect metrics
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Classes: {results['class_names']}")

# Cell 4: Plot all metrics
plot_from_evaluation_results(results_dir, show_plots=True)

# Cell 5: Plot specific visualization
from plot_metrics import plot_confusion_matrix_normalized
plot_confusion_matrix_normalized(
    results['cm'], 
    results['class_names'], 
    format_type='percent'
)
```

## Requirements

Make sure you have the required packages:

```bash
pip install matplotlib seaborn numpy
```

## File Structure

After running `evaluate_model.py`, you should have:

```
evaluation_results/
└── evaluation_20240101_120000/
    ├── detailed_metrics.txt       # ← Used for plotting (simplest!)
    ├── classification_report.json
    ├── confusion_matrix.json
    └── ...
```

The `detailed_metrics.txt` file contains all the data needed for plotting. The script automatically parses this file to extract:
- Accuracy
- Per-class metrics (Precision, Recall, F1-Score, Support)
- Confusion matrix

## Troubleshooting

### "Confusion matrix not available"

If you see this message, it means the parser couldn't extract the confusion matrix from `detailed_metrics.txt`. 

**Solution:** Make sure `evaluate_model.py` completed successfully and generated `detailed_metrics.txt` with the confusion matrix section.

### "Evaluation results not found"

Make sure the path to your evaluation results directory is correct:

```python
import os
results_dir = "evaluation_results/evaluation_20240101_120000"
print(f"Directory exists: {os.path.exists(results_dir)}")
```

### Plots Not Displaying

Make sure you have `%matplotlib inline` at the top of your notebook, or use:

```python
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode
```

## Command Line Usage

You can also use `plot_metrics.py` from the command line:

```bash
# Plot from evaluation results
python plot_metrics.py --results-dir evaluation_results/evaluation_20240101_120000

# Save to custom directory
python plot_metrics.py --results-dir evaluation_results/evaluation_20240101_120000 --save-dir my_plots

# Save only (don't display)
python plot_metrics.py --results-dir evaluation_results/evaluation_20240101_120000 --no-show
```


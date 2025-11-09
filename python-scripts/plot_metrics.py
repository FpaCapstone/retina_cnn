"""
Plot Metrics for Model Evaluation Results

This script creates visualizations for model evaluation metrics.
Can be run standalone or in JupyterLab.

Usage in JupyterLab:
    # Load from saved evaluation results
    from plot_metrics import plot_from_evaluation_results
    plot_from_evaluation_results('evaluation_results/evaluation_20240101_120000')
    
    # Or plot from metrics directly
    from plot_metrics import plot_metrics
    plot_metrics(cm, class_names, precision, recall, f1, save_dir='plots')
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple

# ============================================
# CONFIGURATION
# ============================================
DEFAULT_SAVE_DIR = "plots"
DEFAULT_RESULTS_DIR = "evaluation_results"

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def setup_plot_style():
    """Configure matplotlib and seaborn style."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")

def plot_per_class_metrics(
    class_names: List[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot per-class metrics (Precision, Recall, F1-Score) as bar chart.
    
    Args:
        class_names: List of class names
        precision: Array of precision scores per class
        recall: Array of recall scores per class
        f1: Array of F1 scores per class
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics: Precision, Recall, and F1-Score', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved per-class metrics plot to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_confusion_matrix_absolute(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot confusion matrix with absolute counts.
    
    Args:
        cm: Confusion matrix (2D array)
        class_names: List of class names
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title('Confusion Matrix (Absolute Counts)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved absolute confusion matrix to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_confusion_matrix_normalized(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    format_type: str = 'decimal'
) -> plt.Figure:
    """
    Plot normalized confusion matrix (decimal or percent).
    
    Args:
        cm: Confusion matrix (2D array)
        class_names: List of class names
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
        format_type: 'decimal' (0-1) or 'percent' (0-100%)
    
    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if format_type == 'percent':
        cm_normalized = cm_normalized * 100
        fmt = '.1f'
        cbar_label = 'Percentage (%)'
        title_suffix = 'Percent View'
        vmax = 100
    else:
        fmt = '.3f'
        cbar_label = 'Normalized Value'
        title_suffix = 'Decimal'
        vmax = 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=vmax,
        annot_kws={'fontsize': 10, 'fontweight': 'bold'} if format_type == 'percent' else {},
        ax=ax
    )
    
    ax.set_title(f'Confusion Matrix (Normalized - {title_suffix})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved normalized confusion matrix to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_all_metrics(
    cm: np.ndarray,
    class_names: List[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    save_dir: str = DEFAULT_SAVE_DIR,
    show_plots: bool = False
) -> dict:
    """
    Create all metric visualizations.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        precision: Array of precision scores
        recall: Array of recall scores
        f1: Array of F1 scores
        save_dir: Directory to save plots
        show_plots: Whether to display plots (set False for Jupyter batch processing)
    
    Returns:
        Dictionary with paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = {}
    
    # 1. Per-class metrics
    metrics_path = os.path.join(save_dir, "per_class_metrics.png")
    plot_per_class_metrics(class_names, precision, recall, f1, 
                          save_path=metrics_path, show_plot=show_plots)
    saved_paths['per_class_metrics'] = metrics_path
    
    # 2. Confusion matrix - absolute
    cm_abs_path = os.path.join(save_dir, "confusion_matrix_absolute.png")
    plot_confusion_matrix_absolute(cm, class_names, 
                                  save_path=cm_abs_path, show_plot=show_plots)
    saved_paths['confusion_matrix_absolute'] = cm_abs_path
    
    # 3. Confusion matrix - normalized decimal
    cm_norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix_normalized(cm, class_names, format_type='decimal',
                                    save_path=cm_norm_path, show_plot=show_plots)
    saved_paths['confusion_matrix_normalized'] = cm_norm_path
    
    # 4. Confusion matrix - normalized percent
    cm_percent_path = os.path.join(save_dir, "confusion_matrix_percent.png")
    plot_confusion_matrix_normalized(cm, class_names, format_type='percent',
                                    save_path=cm_percent_path, show_plot=show_plots)
    saved_paths['confusion_matrix_percent'] = cm_percent_path
    
    # 5. Default confusion_matrix.png (same as percent view)
    cm_default_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix_normalized(cm, class_names, format_type='percent',
                                    save_path=cm_default_path, show_plot=show_plots)
    saved_paths['confusion_matrix'] = cm_default_path
    
    print(f"\n✅ All plots saved to: {save_dir}")
    return saved_paths

# ============================================
# LOAD FROM EVALUATION RESULTS
# ============================================

def parse_detailed_metrics(metrics_file: str) -> dict:
    """
    Parse detailed_metrics.txt file to extract all metrics.
    
    Args:
        metrics_file: Path to detailed_metrics.txt
    
    Returns:
        Dictionary with parsed metrics
    """
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    metrics = {
        'accuracy': 0.0,
        'class_names': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'support': [],
        'cm': None
    }
    
    # Parse accuracy
    for line in lines:
        if 'Accuracy:' in line:
            try:
                acc_str = line.split('Accuracy:')[1].strip().split()[0]
                metrics['accuracy'] = float(acc_str)
            except:
                pass
    
    # Parse per-class metrics table
    in_per_class_section = False
    for i, line in enumerate(lines):
        if 'PER-CLASS METRICS' in line:
            in_per_class_section = True
            continue
        
        if in_per_class_section:
            if 'Class' in line and 'Precision' in line:
                continue  # Skip header
            if line.strip() == '' or '=' in line:
                if metrics['class_names']:  # We've finished the table
                    break
                continue
            
            # Parse metric line: "Class Name    Precision    Recall    F1-Score    Support"
            parts = line.split()
            if len(parts) >= 5:
                try:
                    # Class name might have spaces, so we need to be careful
                    # Find where numbers start
                    for j, part in enumerate(parts):
                        try:
                            float(part)
                            # Found first number
                            class_name = ' '.join(parts[:j])
                            precision = float(parts[j])
                            recall = float(parts[j+1])
                            f1 = float(parts[j+2])
                            support = int(parts[j+3])
                            
                            metrics['class_names'].append(class_name)
                            metrics['precision'].append(precision)
                            metrics['recall'].append(recall)
                            metrics['f1'].append(f1)
                            metrics['support'].append(support)
                            break
                        except ValueError:
                            continue
                except:
                    continue
    
    # Parse confusion matrix
    # Format: Fixed-width columns
    # Line 1: Empty (20 chars) + class names (12 chars each)
    # Lines 2+: Class name (20 chars) + numbers (12 chars each)
    in_cm_section = False
    cm_lines = []
    header_skipped = False
    expected_cols = len(metrics['class_names']) if metrics['class_names'] else 0
    
    for i, line in enumerate(lines):
        if 'CONFUSION MATRIX' in line:
            in_cm_section = True
            continue
        
        if in_cm_section:
            if 'Rows = True Labels' in line or line.strip() == '':
                continue
            
            # Skip the header row (first row after "Rows = True Labels")
            if not header_skipped:
                header_skipped = True
                continue
            
            # Check if we've reached the end (next section with =)
            if '=' in line and len(cm_lines) > 0:
                break  # End of confusion matrix
            
            # Parse fixed-width format: class name (20 chars) + numbers (12 chars each)
            if len(line) >= 20:
                try:
                    # Extract class name (first 20 chars, strip whitespace)
                    class_name_part = line[:20].strip()
                    
                    # Extract numbers (starting from position 20, 12 chars each)
                    row_values = []
                    for col_idx in range(expected_cols):
                        start_pos = 20 + (col_idx * 12)
                        end_pos = start_pos + 12
                        if end_pos <= len(line):
                            value_str = line[start_pos:end_pos].strip()
                            try:
                                value = int(value_str)
                                row_values.append(value)
                            except ValueError:
                                # If we can't parse, try 0 or break
                                row_values.append(0)
                        else:
                            break
                    
                    # Only add if we have the correct number of values
                    if len(row_values) == expected_cols:
                        cm_lines.append(row_values)
                    elif len(row_values) > 0:
                        # Try to pad if we're close
                        if len(row_values) < expected_cols:
                            row_values.extend([0] * (expected_cols - len(row_values)))
                            cm_lines.append(row_values)
                except Exception as e:
                    # Fallback: try splitting by whitespace
                    parts = line.split()
                    if len(parts) > 1:
                        row_values = []
                        for part in parts[1:]:  # Skip first part (class name)
                            try:
                                row_values.append(int(part))
                            except ValueError:
                                break
                        if len(row_values) == expected_cols:
                            cm_lines.append(row_values)
                    continue
    
    # Convert to numpy arrays
    if metrics['class_names']:
        metrics['precision'] = np.array(metrics['precision'])
        metrics['recall'] = np.array(metrics['recall'])
        metrics['f1'] = np.array(metrics['f1'])
        metrics['support'] = np.array(metrics['support'])
    
    # Validate and set confusion matrix
    if cm_lines:
        if len(cm_lines) == len(metrics['class_names']):
            # Check if all rows have the correct length
            all_correct_length = all(len(row) == len(metrics['class_names']) for row in cm_lines)
            if all_correct_length:
                metrics['cm'] = np.array(cm_lines)
                print(f"✅ Successfully parsed confusion matrix: {metrics['cm'].shape}")
            else:
                print(f"⚠️  Confusion matrix rows have inconsistent lengths. Expected {len(metrics['class_names'])}, got {[len(row) for row in cm_lines]}")
        else:
            print(f"⚠️  Confusion matrix has {len(cm_lines)} rows but expected {len(metrics['class_names'])} classes")
    else:
        print("⚠️  Could not parse confusion matrix from detailed_metrics.txt")
    
    return metrics

def load_evaluation_results(results_dir: str) -> dict:
    """
    Load metrics from a saved evaluation results directory.
    Reads from detailed_metrics.txt (simplest approach).
    
    Args:
        results_dir: Path to evaluation results directory
    
    Returns:
        Dictionary containing:
        - cm: confusion matrix (numpy array) or None
        - class_names: list of class names
        - precision: precision array
        - recall: recall array
        - f1: F1-score array
        - accuracy: overall accuracy
        - support: support array
    """
    results_path = Path(results_dir)
    
    # Try to load from detailed_metrics.txt (preferred - simplest)
    metrics_file = results_path / "detailed_metrics.txt"
    if metrics_file.exists():
        print(f"✅ Loading metrics from: {metrics_file}")
        metrics = parse_detailed_metrics(str(metrics_file))
        
        # Debug output
        print(f"   Loaded {len(metrics['class_names'])} classes: {metrics['class_names']}")
        print(f"   Confusion matrix available: {metrics['cm'] is not None}")
        if metrics['cm'] is not None:
            print(f"   Confusion matrix shape: {metrics['cm'].shape}")
        
        return {
            'cm': metrics['cm'],
            'class_names': metrics['class_names'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'support': metrics['support']
        }
    
    # Fallback: Try confusion_matrix.json
    cm_json_path = results_path / "confusion_matrix.json"
    if cm_json_path.exists():
        print(f"✅ Loading from confusion_matrix.json: {cm_json_path}")
        with open(cm_json_path, 'r') as f:
            cm_data = json.load(f)
        
        return {
            'cm': np.array(cm_data['confusion_matrix']),
            'class_names': cm_data['class_names'],
            'precision': np.array(cm_data['precision']),
            'recall': np.array(cm_data['recall']),
            'f1': np.array(cm_data['f1']),
            'accuracy': cm_data['accuracy'],
            'support': np.array(cm_data['support'])
        }
    
    # Fallback: Try classification_report.json
    report_path = results_path / "classification_report.json"
    if report_path.exists():
        print(f"⚠️  Loading from classification_report.json (confusion matrix not available)")
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Extract class names (exclude macro/micro avg)
        class_names = [k for k in report.keys() 
                       if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # Extract metrics
        precision = np.array([report[cls]['precision'] for cls in class_names])
        recall = np.array([report[cls]['recall'] for cls in class_names])
        f1 = np.array([report[cls]['f1-score'] for cls in class_names])
        accuracy = report.get('accuracy', 0.0)
        support = np.array([report[cls]['support'] for cls in class_names])
        
        return {
            'cm': None,  # Not available
            'class_names': class_names,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': support
        }
    
    # Nothing found
    raise FileNotFoundError(
        f"Evaluation results not found in: {results_dir}\n"
        f"Expected file: detailed_metrics.txt\n"
        f"Make sure you've run evaluate_model.py first."
    )

def plot_from_evaluation_results(
    results_dir: str,
    save_dir: Optional[str] = None,
    show_plots: bool = True
) -> dict:
    """
    Load evaluation results and create all plots.
    
    Args:
        results_dir: Path to evaluation results directory
        save_dir: Optional directory to save plots (default: same as results_dir)
        show_plots: Whether to display plots
    
    Returns:
        Dictionary with paths to saved plots
    """
    print(f"📥 Loading evaluation results from: {results_dir}")
    results = load_evaluation_results(results_dir)
    
    if save_dir is None:
        save_dir = results_dir
    
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = {}
    
    # Plot per-class metrics (always available)
    print("\n📊 Creating per-class metrics plot...")
    metrics_path = os.path.join(save_dir, "per_class_metrics.png")
    plot_per_class_metrics(
        results['class_names'],
        results['precision'],
        results['recall'],
        results['f1'],
        save_path=metrics_path,
        show_plot=show_plots
    )
    saved_paths['per_class_metrics'] = metrics_path
    
    # Plot confusion matrices if available
    if results['cm'] is not None:
        print("\n📊 Creating confusion matrix plots...")
        
        # Absolute
        cm_abs_path = os.path.join(save_dir, "confusion_matrix_absolute.png")
        plot_confusion_matrix_absolute(
            results['cm'],
            results['class_names'],
            save_path=cm_abs_path,
            show_plot=show_plots
        )
        saved_paths['confusion_matrix_absolute'] = cm_abs_path
        
        # Normalized decimal
        cm_norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
        plot_confusion_matrix_normalized(
            results['cm'],
            results['class_names'],
            format_type='decimal',
            save_path=cm_norm_path,
            show_plot=show_plots
        )
        saved_paths['confusion_matrix_normalized'] = cm_norm_path
        
        # Normalized percent
        cm_percent_path = os.path.join(save_dir, "confusion_matrix_percent.png")
        plot_confusion_matrix_normalized(
            results['cm'],
            results['class_names'],
            format_type='percent',
            save_path=cm_percent_path,
            show_plot=show_plots
        )
        saved_paths['confusion_matrix_percent'] = cm_percent_path
        
        # Default confusion_matrix.png (same as percent view)
        cm_default_path = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix_normalized(
            results['cm'],
            results['class_names'],
            format_type='percent',
            save_path=cm_default_path,
            show_plot=show_plots
        )
        saved_paths['confusion_matrix'] = cm_default_path
    else:
        print("\n⚠️  Confusion matrix not available in results.")
        print("   Only per-class metrics plotted.")
    
    print(f"\n✅ All available plots saved to: {save_dir}")
    return saved_paths

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: List[str],
    save_dir: str = DEFAULT_SAVE_DIR,
    show_plots: bool = True
) -> dict:
    """
    Plot all confusion matrix visualizations (absolute, normalized decimal, normalized percent).
    
    Convenience function to plot only confusion matrices without other metrics.
    
    Usage in Jupyter:
        from plot_metrics import plot_confusion_matrices
        import numpy as np
        
        cm = np.array([[85, 3, 2], [4, 72, 5], [2, 4, 81]])
        class_names = ['Normal', 'Uveitis', 'Cataract']
        
        plot_confusion_matrices(cm, class_names)
    
    Args:
        cm: Confusion matrix (2D numpy array)
        class_names: List of class names
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    
    Returns:
        Dictionary with paths to saved plots
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = {}
    
    print("📊 Creating confusion matrix plots...")
    
    # Absolute counts
    cm_abs_path = os.path.join(save_dir, "confusion_matrix_absolute.png")
    plot_confusion_matrix_absolute(
        cm, class_names, 
        save_path=cm_abs_path, 
        show_plot=show_plots
    )
    saved_paths['confusion_matrix_absolute'] = cm_abs_path
    
    # Normalized decimal
    cm_norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix_normalized(
        cm, class_names, 
        format_type='decimal',
        save_path=cm_norm_path, 
        show_plot=show_plots
    )
    saved_paths['confusion_matrix_normalized'] = cm_norm_path
    
    # Normalized percent
    cm_percent_path = os.path.join(save_dir, "confusion_matrix_percent.png")
    plot_confusion_matrix_normalized(
        cm, class_names, 
        format_type='percent',
        save_path=cm_percent_path, 
        show_plot=show_plots
    )
    saved_paths['confusion_matrix_percent'] = cm_percent_path
    
    # Default confusion_matrix.png (same as percent view)
    cm_default_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix_normalized(
        cm, class_names, 
        format_type='percent',
        save_path=cm_default_path, 
        show_plot=show_plots
    )
    saved_paths['confusion_matrix'] = cm_default_path
    
    print(f"✅ All confusion matrix plots saved to: {save_dir}")
    return saved_paths

# ============================================
# JUPYTER NOTEBOOK HELPERS
# ============================================

def plot_metrics(
    cm: np.ndarray,
    class_names: List[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    save_dir: str = DEFAULT_SAVE_DIR,
    show_plots: bool = True
) -> dict:
    """
    Main function to plot all metrics. Jupyter-friendly.
    
    Usage in Jupyter:
        from plot_metrics import plot_metrics
        import numpy as np
        
        # Your metrics
        cm = np.array([[85, 5, 2], [3, 90, 1], [1, 2, 88]])
        class_names = ['Normal', 'Uveitis', 'Cataract']
        precision = np.array([0.95, 0.94, 0.97])
        recall = np.array([0.92, 0.96, 0.97])
        f1 = np.array([0.94, 0.95, 0.97])
        
        # Plot all
        plot_metrics(cm, class_names, precision, recall, f1)
    
    Args:
        cm: Confusion matrix (2D numpy array)
        class_names: List of class names
        precision: Array of precision scores per class
        recall: Array of recall scores per class
        f1: Array of F1 scores per class
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    
    Returns:
        Dictionary with paths to saved plots
    """
    return plot_all_metrics(cm, class_names, precision, recall, f1, 
                           save_dir=save_dir, show_plots=show_plots)

# ============================================
# MAIN / CLI
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot metrics from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from evaluation results directory
  python plot_metrics.py --results-dir evaluation_results/evaluation_20240101_120000
  
  # Plot and save to custom directory
  python plot_metrics.py --results-dir evaluation_results/evaluation_20240101_120000 --save-dir my_plots
        """
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to evaluation results directory"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as results-dir)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (save only)"
    )
    
    args = parser.parse_args()
    
    if args.results_dir:
        plot_from_evaluation_results(
            args.results_dir,
            save_dir=args.save_dir,
            show_plots=not args.no_show
        )
    else:
        print("Please provide --results-dir or use in JupyterLab")
        print("\nJupyterLab usage:")
        print("  from plot_metrics import plot_metrics, plot_from_evaluation_results")
        print("  plot_from_evaluation_results('evaluation_results/evaluation_20240101_120000')")


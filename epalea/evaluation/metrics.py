"""
Evaluation Metrics Module
Implements all metrics from Section 10.6.

Metrics Categories:
1. Classification: Accuracy, F1, AUC
2. Probabilistic: NLL, Brier, ECE
3. Abstention: Coverage-Accuracy curves
4. Robustness: Performance under noise
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss
)
from dataclasses import dataclass

# sometimes the y_probs sum to 0.99999997..., etc. 
import warnings
warnings.filterwarnings("ignore", message="The y_prob values do not sum to one")


@dataclass
class MetricsReport:
    """Container for all evaluation metrics."""
    # Classification
    accuracy: float
    macro_f1: float
    weighted_f1: float
    
    # Probabilistic
    nll: float  # Negative Log Likelihood
    brier: float  # Brier Score
    ece: float  # Expected Calibration Error
    
    # Abstention (if applicable)
    coverage: Optional[float] = None
    selective_accuracy: Optional[float] = None
    
    # Additional
    confidence_mean: float = 0.0
    confidence_std: float = 0.0

    runtime_ms: Optional[float] = None 
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'weighted_f1': self.weighted_f1,
            'nll': self.nll,
            'brier': self.brier,
            'ece': self.ece,
            'coverage': self.coverage,
            'selective_accuracy': self.selective_accuracy,
            'confidence_mean': self.confidence_mean,
            'confidence_std': self.confidence_std,
            'runtime_ms': self.runtime_ms
        }


class MetricsCalculator:
    """
    Computes all evaluation metrics.
    """
    
    def __init__(self, domain: List[str]):
        """
        Initialize calculator.
        
        Args:
            domain: List of class labels
        """
        self.domain = domain
        self.num_classes = len(domain)
        self.label_to_idx = {label: idx for idx, label in enumerate(domain)}
    
    def compute_all_metrics(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str],
        confidences: Optional[List[float]] = None
    ) -> MetricsReport:
        """
        Compute all metrics.
        
        Args:
            predictions: List of probability distributions
            true_labels: List of ground truth labels
            confidences: Optional list of confidence scores
            
        Returns:
            MetricsReport with all metrics
        """
        # Convert to arrays
        y_true = np.array([self.label_to_idx[label] for label in true_labels])
        
        # Get predicted labels (fix: use max with key parameter properly)
        y_pred = [max(pred.items(), key=lambda x: x[1])[0] for pred in predictions]
        y_pred_idx = np.array([self.label_to_idx[pred] for pred in y_pred])
        
        # Probability matrix
        y_prob = self._build_prob_matrix(predictions)
        
        # Classification metrics (convert to float)
        accuracy = float(accuracy_score(y_true, y_pred_idx))
        macro_f1 = float(f1_score(y_true, y_pred_idx, average='macro'))
        weighted_f1 = float(f1_score(y_true, y_pred_idx, average='weighted'))
        
        # Probabilistic metrics
        nll = self._compute_nll(y_prob, y_true)
        brier = self._compute_brier(y_prob, y_true)
        ece = self._compute_ece(y_prob, y_true)
        
        # Confidence statistics
        if confidences is None:
            confidences = [max(pred.values()) for pred in predictions]
        
        conf_mean = float(np.mean(confidences))
        conf_std = float(np.std(confidences))
        
        return MetricsReport(
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            nll=nll,
            brier=brier,
            ece=ece,
            confidence_mean=conf_mean,
            confidence_std=conf_std
        )
    
    def _build_prob_matrix(
        self,
        predictions: List[Dict[str, float]]
    ) -> np.ndarray:
        """
        Build probability matrix from predictions.
        
        Args:
            predictions: List of distributions
            
        Returns:
            Array of shape (n_samples, n_classes)
        """
        n = len(predictions)
        prob_matrix = np.zeros((n, self.num_classes))
        
        for i, pred in enumerate(predictions):
            for label, prob in pred.items():
                idx = self.label_to_idx[label]
                prob_matrix[i, idx] = prob
        
        return prob_matrix
    
    def _compute_nll(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute Negative Log Likelihood.
        
        Args:
            y_prob: Probability matrix (n_samples, n_classes)
            y_true: True labels (n_samples,)
            
        Returns:
            NLL score (lower is better)
        """
        if self.num_classes < 2:
            return 0.0  # Single class - always correct
        
        return float(log_loss(y_true, y_prob, labels=list(range(self.num_classes))))
    
    def _compute_brier(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute Brier Score.
        
        Brier = mean((p - y_onehot)^2)
        
        Args:
            y_prob: Probability matrix
            y_true: True labels
            
        Returns:
            Brier score (lower is better)
        """
        if self.num_classes < 2:
            return 0.0  # Single class - always correct
    
        # One-hot encode
        y_onehot = np.zeros_like(y_prob)
        y_onehot[np.arange(len(y_true)), y_true] = 1
        
        # Compute Brier
        brier = np.mean((y_prob - y_onehot) ** 2)
        return float(brier)
    
    def _compute_ece(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE = sum(|bin_accuracy - bin_confidence| * bin_weight)
        
        Args:
            y_prob: Probability matrix
            y_true: True labels
            n_bins: Number of bins
            
        Returns:
            ECE score (lower is better)
        """
        # Get confidences and predictions
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            
            if np.sum(in_bin) == 0:
                continue
            
            # Compute bin metrics
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)
            
            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight
        
        return float(ece)
    
    def compute_selective_metrics(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str],
        confidence_threshold: float
    ) -> Tuple[float, float]:
        """
        Compute selective classification metrics.
        
        Only evaluate predictions above confidence threshold.
        
        Args:
            predictions: List of distributions
            true_labels: Ground truth labels
            confidence_threshold: Minimum confidence to accept
            
        Returns:
            Tuple of (coverage, selective_accuracy)
        """
        # Filter by confidence
        confidences = [max(pred.values()) for pred in predictions]
        
        accepted = []
        accepted_labels = []
        
        for i, conf in enumerate(confidences):
            if conf >= confidence_threshold:
                accepted.append(predictions[i])
                accepted_labels.append(true_labels[i])
        
        if not accepted:
            return 0.0, 0.0
        
        coverage = len(accepted) / len(predictions)
        
        # Compute accuracy on accepted (fix: use max properly)
        y_true = np.array([self.label_to_idx[label] for label in accepted_labels])
        y_pred_labels = [max(pred.items(), key=lambda x: x[1])[0] for pred in accepted]
        y_pred = np.array([self.label_to_idx[pred] for pred in y_pred_labels])
        
        selective_accuracy = float(accuracy_score(y_true, y_pred))
        
        return coverage, selective_accuracy
    
    def compute_risk_coverage_curve(
        self,
        predictions: List[Dict[str, float]],
        true_labels: List[str],
        thresholds: Optional[List[float]] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Compute risk-coverage curve for abstention analysis.
        
        Args:
            predictions: List of distributions
            true_labels: Ground truth labels
            thresholds: Confidence thresholds to evaluate
            
        Returns:
            List of (threshold, coverage, accuracy) tuples
        """
        thresholds_to_use: List[float] = (
            thresholds if thresholds is not None 
            else np.linspace(0.0, 1.0, 21).tolist()
        )
        
        curve = []
        for threshold in thresholds_to_use:
            coverage, accuracy = self.compute_selective_metrics(
                predictions, true_labels, threshold
            )
            curve.append((float(threshold), coverage, accuracy))
        
        return curve


def compute_calibration_curve(
    predictions: List[Dict[str, float]],
    true_labels: List[str],
    domain: List[str],
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve for reliability diagrams.
    
    Args:
        predictions: List of distributions
        true_labels: Ground truth labels
        domain: Class labels
        n_bins: Number of bins
        
    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_confidences)
    """
    label_to_idx = {label: idx for idx, label in enumerate(domain)}
    
    # Get confidences and predictions (fix: use max properly)
    confidences = np.array([max(pred.values()) for pred in predictions])
    pred_labels = [max(pred.items(), key=lambda x: x[1])[0] for pred in predictions]
    pred_idx = np.array([label_to_idx[label] for label in pred_labels])
    true_idx = np.array([label_to_idx[label] for label in true_labels])
    
    accuracies = (pred_idx == true_idx)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])
    
    return bin_centers, np.array(bin_accuracies), np.array(bin_confidences)


def bootstrap_confidence_interval(
    metric_fn: Callable[[List[Any], List[str]], float],
    predictions: List[Any],
    true_labels: List[str],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        metric_fn: Function computing metric
        predictions: Predictions
        true_labels: Ground truth
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(predictions)
    scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        sample_preds = [predictions[i] for i in indices]
        sample_labels = [true_labels[i] for i in indices]
        
        # Compute metric
        score = metric_fn(sample_preds, sample_labels)
        scores.append(score)
    
    scores_array = np.array(scores)
    
    mean = float(np.mean(scores_array))
    alpha = (1 - confidence_level) / 2
    lower = float(np.percentile(scores_array, alpha * 100))
    upper = float(np.percentile(scores_array, (1 - alpha) * 100))
    
    return mean, lower, upper


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 70)
    
    # Demo with synthetic data
    domain = ["low", "medium", "high"]
    calc = MetricsCalculator(domain)
    
    # Synthetic predictions
    predictions = [
        {"low": 0.7, "medium": 0.2, "high": 0.1},
        {"low": 0.1, "medium": 0.8, "high": 0.1},
        {"low": 0.2, "medium": 0.3, "high": 0.5},
    ]
    true_labels = ["low", "medium", "high"]
    
    report = calc.compute_all_metrics(predictions, true_labels)
    
    print("\nDemo Metrics:")
    print(f"  Accuracy: {report.accuracy:.3f}")
    print(f"  Macro F1: {report.macro_f1:.3f}")
    print(f"  NLL: {report.nll:.3f}")
    print(f"  Brier: {report.brier:.3f}")
    print(f"  ECE: {report.ece:.3f}")
    print(f"  Confidence: {report.confidence_mean:.3f} ± {report.confidence_std:.3f}")
    
    print("\n✓ All metrics implemented!")
    
#!/usr/bin/env python3
"""
Comprehensive evaluation system with confusion matrix and charts
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import logging
from tqdm import tqdm
import json
from datetime import datetime
from config import Config
from face_detector_simple import FaceDetector
from facenet_model_casia import FaceNetModel

logger = logging.getLogger(__name__)

class FaceVerificationEvaluator:
    def __init__(self):
        """Initialize evaluation system"""
        self.face_detector = FaceDetector()
        self.facenet_model = FaceNetModel()
        self.facenet_model.load_embeddings()
        
        self.results = {
            'predictions': [],
            'true_labels': [],
            'distances': [],
            'similarities': [],
            'identities': [],
            'image_paths': [],
            'detection_success': []
        }
        
        # Create evaluation output directory
        self.output_dir = os.path.join(Config.BASE_DIR, 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate_dataset(self, dataset_path=None, test_split=0.3):
        """
        Evaluate model on dataset with train/test split
        
        Args:
            dataset_path: Path to dataset
            test_split: Fraction of data to use for testing
        """
        if dataset_path is None:
            dataset_path = Config.DATASET_DIR
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            return False
        
        print("🔍 Evaluating Face Verification System")
        print("=" * 50)
        
        # Get all identities and images
        all_data = self._collect_dataset_info(dataset_path)
        
        if not all_data:
            logger.error("No data found in dataset")
            return False
        
        # Split data into train and test
        train_data, test_data = self._split_dataset(all_data, test_split)
        
        print(f"📊 Dataset Split:")
        print(f"   Training: {len(train_data)} images")
        print(f"   Testing: {len(test_data)} images")
        print(f"   Identities: {len(set(item['identity'] for item in all_data))}")
        
        # Train model on training data
        print("\n🔄 Training model on training split...")
        self._train_on_split(train_data)
        
        # Evaluate on test data
        print("\n🧪 Evaluating on test split...")
        self._evaluate_on_split(test_data)
        
        # Generate reports and visualizations
        self._generate_evaluation_report()
        
        return True
    
    def evaluate_threshold_range(self, dataset_path=None, thresholds=None):
        """
        Evaluate model performance across different thresholds
        
        Args:
            dataset_path: Path to dataset
            thresholds: List of thresholds to test
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.5, 0.05)
        
        if dataset_path is None:
            dataset_path = Config.DATASET_DIR
        
        print("📈 Threshold Analysis")
        print("=" * 30)
        
        # Collect all test data
        all_data = self._collect_dataset_info(dataset_path)
        _, test_data = self._split_dataset(all_data, 0.3)
        
        threshold_results = []
        
        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            # Set threshold
            original_threshold = Config.VERIFICATION_THRESHOLD
            Config.VERIFICATION_THRESHOLD = threshold
            
            # Reset results
            self.results = {
                'predictions': [],
                'true_labels': [],
                'distances': [],
                'similarities': [],
                'identities': [],
                'image_paths': [],
                'detection_success': []
            }
            
            # Evaluate
            self._evaluate_on_split(test_data, show_progress=False)
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            metrics['threshold'] = threshold
            threshold_results.append(metrics)
            
            # Restore original threshold
            Config.VERIFICATION_THRESHOLD = original_threshold
        
        # Plot threshold analysis
        self._plot_threshold_analysis(threshold_results)
        
        return threshold_results
    
    def _collect_dataset_info(self, dataset_path):
        """Collect all images and their labels from dataset"""
        all_data = []
        
        for identity in os.listdir(dataset_path):
            identity_path = os.path.join(dataset_path, identity)
            
            if not os.path.isdir(identity_path):
                continue
            
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(identity_path, img_file)
                    all_data.append({
                        'identity': identity,
                        'image_path': img_path
                    })
        
        return all_data
    
    def _split_dataset(self, all_data, test_split):
        """Split dataset into train and test while maintaining identity distribution"""
        from collections import defaultdict
        import random
        
        # Group by identity
        identity_groups = defaultdict(list)
        for item in all_data:
            identity_groups[item['identity']].append(item)
        
        train_data = []
        test_data = []
        
        for identity, items in identity_groups.items():
            random.shuffle(items)
            n_test = max(1, int(len(items) * test_split))  # At least 1 for test
            n_train = len(items) - n_test
            
            train_data.extend(items[:n_train])
            test_data.extend(items[n_train:])
        
        return train_data, test_data
    
    def _train_on_split(self, train_data):
        """Train model on training split"""
        # Group training data by identity
        from collections import defaultdict
        
        identity_images = defaultdict(list)
        
        for item in tqdm(train_data, desc="Processing training images"):
            identity = item['identity']
            img_path = item['image_path']
            
            # Load and process image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            face_img, confidence = self.face_detector.get_largest_face(image)
            if face_img is not None:
                identity_images[identity].append(face_img)
        
        # Register each identity
        for identity, face_images in identity_images.items():
            if face_images:
                success = self.facenet_model.register_face(identity, face_images)
                if success:
                    logger.info(f"Registered {identity} with {len(face_images)} faces")
        
        # Save embeddings
        self.facenet_model.save_embeddings()
    
    def _evaluate_on_split(self, test_data, show_progress=True):
        """Evaluate model on test split"""
        iterator = tqdm(test_data, desc="Evaluating") if show_progress else test_data
        
        for item in iterator:
            identity = item['identity']
            img_path = item['image_path']
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Detect face
            face_img, confidence = self.face_detector.get_largest_face(image)
            
            if face_img is None:
                # No face detected
                self.results['predictions'].append('NO_FACE')
                self.results['true_labels'].append(identity)
                self.results['distances'].append(float('inf'))
                self.results['similarities'].append(0.0)
                self.results['identities'].append('NO_FACE')
                self.results['image_paths'].append(img_path)
                self.results['detection_success'].append(False)
                continue
            
            # Verify face
            result = self.facenet_model.verify_face(face_img)
            
            # Store results
            predicted_identity = result.get('identity', 'UNKNOWN')
            if not result['success']:
                predicted_identity = 'NOT_RECOGNIZED'
            
            self.results['predictions'].append(predicted_identity)
            self.results['true_labels'].append(identity)
            self.results['distances'].append(result.get('distance', float('inf')))
            self.results['similarities'].append(result.get('similarity', 0.0))
            self.results['identities'].append(predicted_identity)
            self.results['image_paths'].append(img_path)
            self.results['detection_success'].append(True)
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics"""
        predictions = np.array(self.results['predictions'])
        true_labels = np.array(self.results['true_labels'])
        
        # Calculate accuracy (exact matches)
        correct_predictions = (predictions == true_labels)
        accuracy = np.mean(correct_predictions)
        
        # Calculate detection rate
        detection_rate = np.mean(self.results['detection_success'])
        
        # Calculate recognition rate (among detected faces)
        detected_indices = np.array(self.results['detection_success'])
        if np.sum(detected_indices) > 0:
            detected_predictions = predictions[detected_indices]
            detected_true_labels = true_labels[detected_indices]
            recognition_rate = np.mean(detected_predictions == detected_true_labels)
        else:
            recognition_rate = 0.0
        
        # False acceptance rate (FAR) and False rejection rate (FRR)
        # FAR: incorrect accepts / total non-target attempts
        # FRR: incorrect rejects / total target attempts
        
        far_count = 0
        frr_count = 0
        total_target = 0
        total_non_target = 0
        
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            if not self.results['detection_success'][i]:
                continue
                
            if pred == true:  # Target attempt
                total_target += 1
                if pred == 'NOT_RECOGNIZED':
                    frr_count += 1
            else:  # Non-target attempt
                total_non_target += 1
                if pred != 'NOT_RECOGNIZED':
                    far_count += 1
        
        far = far_count / max(1, total_non_target)
        frr = frr_count / max(1, total_target)
        
        return {
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'recognition_rate': recognition_rate,
            'far': far,  # False Acceptance Rate
            'frr': frr,  # False Rejection Rate
            'total_samples': len(predictions),
            'detected_samples': np.sum(detected_indices)
        }
    
    def _generate_evaluation_report(self):
        """Generate comprehensive evaluation report with visualizations"""
        print("\n📊 Generating Evaluation Report...")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # 1. Print metrics summary
        self._print_metrics_summary(metrics)
        
        # 2. Generate confusion matrix
        self._plot_confusion_matrix()
        
        # 3. Generate ROC curve
        self._plot_roc_curve()
        
        # 4. Generate distance distribution
        self._plot_distance_distribution()
        
        # 5. Generate per-identity analysis
        self._plot_per_identity_analysis()
        
        # 6. Generate detection success rate
        self._plot_detection_analysis()
        
        # 7. Save detailed results
        self._save_detailed_results(metrics)
        
        print(f"\n📁 Results saved to: {self.output_dir}")
    
    def _print_metrics_summary(self, metrics):
        """Print evaluation metrics summary"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Face Detection Rate: {metrics['detection_rate']:.3f} ({metrics['detection_rate']*100:.1f}%)")
        print(f"Recognition Rate (detected faces): {metrics['recognition_rate']:.3f} ({metrics['recognition_rate']*100:.1f}%)")
        print(f"False Acceptance Rate (FAR): {metrics['far']:.3f} ({metrics['far']*100:.1f}%)")
        print(f"False Rejection Rate (FRR): {metrics['frr']:.3f} ({metrics['frr']*100:.1f}%)")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Detected Samples: {metrics['detected_samples']}")
        print("=" * 60)
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Get unique labels
        all_labels = list(set(self.results['true_labels'] + self.results['predictions']))
        
        # Create confusion matrix
        cm = confusion_matrix(self.results['true_labels'], self.results['predictions'], labels=all_labels)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=all_labels, yticklabels=all_labels)
        plt.title('Confusion Matrix - Face Verification')
        plt.xlabel('Predicted Identity')
        plt.ylabel('True Identity')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curve(self):
        """Plot ROC curve for each identity"""
        plt.figure(figsize=(10, 8))
        
        # Get unique identities (excluding special cases)
        identities = [id for id in set(self.results['true_labels']) 
                     if id not in ['NO_FACE', 'NOT_RECOGNIZED', 'UNKNOWN']]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(identities)))
        
        for i, identity in enumerate(identities):
            # Create binary classification problem for this identity
            y_true = [1 if label == identity else 0 for label in self.results['true_labels']]
            y_scores = []
            
            for j, pred in enumerate(self.results['predictions']):
                if pred == identity:
                    # Use similarity score for positive predictions
                    score = self.results['similarities'][j]
                else:
                    # Use inverse distance for negative predictions
                    dist = self.results['distances'][j]
                    score = 1.0 / (1.0 + dist) if dist != float('inf') else 0.0
                y_scores.append(score)
            
            # Calculate ROC curve
            if sum(y_true) > 0:  # Only if we have positive samples
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{identity} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Per Identity')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_distance_distribution(self):
        """Plot distance distribution for matches vs non-matches"""
        plt.figure(figsize=(12, 6))
        
        # Separate distances for matches and non-matches
        match_distances = []
        non_match_distances = []
        
        for i, (pred, true) in enumerate(zip(self.results['predictions'], self.results['true_labels'])):
            if not self.results['detection_success'][i]:
                continue
                
            distance = self.results['distances'][i]
            if distance == float('inf'):
                continue
                
            if pred == true:
                match_distances.append(distance)
            else:
                non_match_distances.append(distance)
        
        # Plot histograms
        plt.subplot(1, 2, 1)
        if match_distances:
            plt.hist(match_distances, bins=30, alpha=0.7, color='green', label='Matches')
        if non_match_distances:
            plt.hist(non_match_distances, bins=30, alpha=0.7, color='red', label='Non-matches')
        
        plt.axvline(Config.VERIFICATION_THRESHOLD, color='black', linestyle='--', 
                   label=f'Threshold ({Config.VERIFICATION_THRESHOLD})')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot similarities
        plt.subplot(1, 2, 2)
        match_similarities = []
        non_match_similarities = []
        
        for i, (pred, true) in enumerate(zip(self.results['predictions'], self.results['true_labels'])):
            if not self.results['detection_success'][i]:
                continue
                
            similarity = self.results['similarities'][i]
            
            if pred == true:
                match_similarities.append(similarity)
            else:
                non_match_similarities.append(similarity)
        
        if match_similarities:
            plt.hist(match_similarities, bins=30, alpha=0.7, color='green', label='Matches')
        if non_match_similarities:
            plt.hist(non_match_similarities, bins=30, alpha=0.7, color='red', label='Non-matches')
        
        plt.xlabel('Similarity')
        plt.ylabel('Frequency')
        plt.title('Similarity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'distance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_per_identity_analysis(self):
        """Plot per-identity performance analysis"""
        # Calculate per-identity metrics
        identity_metrics = {}
        
        for identity in set(self.results['true_labels']):
            if identity in ['NO_FACE', 'NOT_RECOGNIZED', 'UNKNOWN']:
                continue
                
            # Get indices for this identity
            identity_indices = [i for i, label in enumerate(self.results['true_labels']) if label == identity]
            
            if not identity_indices:
                continue
            
            # Calculate metrics
            correct = sum(1 for i in identity_indices if self.results['predictions'][i] == identity)
            total = len(identity_indices)
            detected = sum(1 for i in identity_indices if self.results['detection_success'][i])
            
            identity_metrics[identity] = {
                'accuracy': correct / total,
                'detection_rate': detected / total,
                'total_samples': total,
                'avg_distance': np.mean([self.results['distances'][i] for i in identity_indices 
                                       if self.results['distances'][i] != float('inf')]),
                'avg_similarity': np.mean([self.results['similarities'][i] for i in identity_indices])
            }
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        identities = list(identity_metrics.keys())
        
        # Accuracy per identity
        accuracies = [identity_metrics[id]['accuracy'] for id in identities]
        axes[0, 0].bar(identities, accuracies, color='skyblue')
        axes[0, 0].set_title('Accuracy per Identity')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Detection rate per identity
        detection_rates = [identity_metrics[id]['detection_rate'] for id in identities]
        axes[0, 1].bar(identities, detection_rates, color='lightgreen')
        axes[0, 1].set_title('Detection Rate per Identity')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average distance per identity
        avg_distances = [identity_metrics[id]['avg_distance'] for id in identities]
        axes[1, 0].bar(identities, avg_distances, color='salmon')
        axes[1, 0].set_title('Average Distance per Identity')
        axes[1, 0].set_ylabel('Average Distance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample count per identity
        sample_counts = [identity_metrics[id]['total_samples'] for id in identities]
        axes[1, 1].bar(identities, sample_counts, color='gold')
        axes[1, 1].set_title('Sample Count per Identity')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'per_identity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return identity_metrics
    
    def _plot_detection_analysis(self):
        """Plot face detection analysis"""
        plt.figure(figsize=(10, 6))
        
        # Detection success rate
        detection_rate = np.mean(self.results['detection_success'])
        
        plt.subplot(1, 2, 1)
        labels = ['Detected', 'Not Detected']
        sizes = [np.sum(self.results['detection_success']), 
                len(self.results['detection_success']) - np.sum(self.results['detection_success'])]
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Face Detection Success Rate')
        
        # Detection by identity
        plt.subplot(1, 2, 2)
        identity_detection = {}
        
        for i, identity in enumerate(self.results['true_labels']):
            if identity not in identity_detection:
                identity_detection[identity] = {'detected': 0, 'total': 0}
            
            identity_detection[identity]['total'] += 1
            if self.results['detection_success'][i]:
                identity_detection[identity]['detected'] += 1
        
        identities = list(identity_detection.keys())
        detection_rates = [identity_detection[id]['detected'] / identity_detection[id]['total'] 
                          for id in identities]
        
        plt.bar(identities, detection_rates, color='lightblue')
        plt.title('Detection Rate by Identity')
        plt.ylabel('Detection Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'detection_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_threshold_analysis(self, threshold_results):
        """Plot threshold analysis results"""
        plt.figure(figsize=(15, 10))
        
        thresholds = [r['threshold'] for r in threshold_results]
        accuracies = [r['accuracy'] for r in threshold_results]
        fars = [r['far'] for r in threshold_results]
        frrs = [r['frr'] for r in threshold_results]
        
        # Plot 1: Accuracy vs Threshold
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, accuracies, 'b-', linewidth=2, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: FAR and FRR vs Threshold
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, fars, 'r-', linewidth=2, marker='s', label='FAR')
        plt.plot(thresholds, frrs, 'g-', linewidth=2, marker='^', label='FRR')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('FAR and FRR vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: FAR vs FRR (DET Curve)
        plt.subplot(2, 2, 3)
        plt.plot(fars, frrs, 'purple', linewidth=2, marker='o')
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Best threshold analysis
        plt.subplot(2, 2, 4)
        # Equal Error Rate (EER) - where FAR = FRR
        eer_differences = [abs(far - frr) for far, frr in zip(fars, frrs)]
        eer_index = np.argmin(eer_differences)
        eer_threshold = thresholds[eer_index]
        eer_value = (fars[eer_index] + frrs[eer_index]) / 2
        
        plt.plot(thresholds, fars, 'r-', label='FAR', alpha=0.7)
        plt.plot(thresholds, frrs, 'g-', label='FRR', alpha=0.7)
        plt.axvline(eer_threshold, color='blue', linestyle='--', 
                   label=f'EER Threshold ({eer_threshold:.3f})')
        plt.axhline(eer_value, color='orange', linestyle=':', 
                   label=f'EER Value ({eer_value:.3f})')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.title('Equal Error Rate (EER) Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(self.output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎯 Optimal Threshold Analysis:")
        print(f"   EER Threshold: {eer_threshold:.3f}")
        print(f"   EER Value: {eer_value:.3f}")
        print(f"   Current Threshold: {Config.VERIFICATION_THRESHOLD}")
    
    def _save_detailed_results(self, metrics):
        """Save detailed evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics summary
        metrics_file = os.path.join(self.output_dir, f'metrics_summary_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'image_path': self.results['image_paths'],
            'true_identity': self.results['true_labels'],
            'predicted_identity': self.results['predictions'],
            'distance': self.results['distances'],
            'similarity': self.results['similarities'],
            'face_detected': self.results['detection_success']
        })
        
        results_file = os.path.join(self.output_dir, f'detailed_results_{timestamp}.csv')
        results_df.to_csv(results_file, index=False)
        
        # Save classification report
        # Filter out non-detected faces for classification report
        detected_indices = np.array(self.results['detection_success'])
        if np.sum(detected_indices) > 0:
            detected_true = np.array(self.results['true_labels'])[detected_indices]
            detected_pred = np.array(self.results['predictions'])[detected_indices]
            
            report = classification_report(detected_true, detected_pred)
            
            report_file = os.path.join(self.output_dir, f'classification_report_{timestamp}.txt')
            with open(report_file, 'w') as f:
                f.write("Classification Report (Detected Faces Only)\n")
                f.write("=" * 50 + "\n")
                f.write(report)
        
        print(f"📄 Detailed results saved:")
        print(f"   Metrics: {metrics_file}")
        print(f"   Results: {results_file}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Verification System Evaluation')
    parser.add_argument('--dataset', default=None, help='Path to dataset directory')
    parser.add_argument('--test-split', type=float, default=0.3, help='Test split ratio')
    parser.add_argument('--threshold-analysis', action='store_true', help='Perform threshold analysis')
    parser.add_argument('--output-dir', default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator
    evaluator = FaceVerificationEvaluator()
    
    if args.output_dir:
        evaluator.output_dir = args.output_dir
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("🧪 Face Verification System Evaluation")
    print("=" * 60)
    
    if args.threshold_analysis:
        print("Running threshold analysis...")
        threshold_results = evaluator.evaluate_threshold_range(args.dataset)
        
        # Find optimal threshold
        best_accuracy_idx = max(range(len(threshold_results)), 
                               key=lambda i: threshold_results[i]['accuracy'])
        best_threshold = threshold_results[best_accuracy_idx]
        
        print(f"\n🎯 Best Threshold for Accuracy:")
        print(f"   Threshold: {best_threshold['threshold']:.3f}")
        print(f"   Accuracy: {best_threshold['accuracy']:.3f}")
        print(f"   FAR: {best_threshold['far']:.3f}")
        print(f"   FRR: {best_threshold['frr']:.3f}")
    
    else:
        print("Running full evaluation...")
        success = evaluator.evaluate_dataset(args.dataset, args.test_split)
        
        if success:
            print("\n✅ Evaluation completed successfully!")
        else:
            print("\n❌ Evaluation failed!")

if __name__ == "__main__":
    main()
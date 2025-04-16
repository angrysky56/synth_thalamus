"""
Parameter sweep utilities for the Synthetic Thalamus

This module provides tools for systematically evaluating different parameter
configurations for the Synthetic Thalamus, particularly for the enhanced
semantic phase generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import os
import json

from core.phase_generator import EnhancedSemanticPhaseGenerator, PhaseAnalyzer, contrastive_loss


class PhaseGeneratorSweep:
    """
    Parameter sweep for the Enhanced Semantic Phase Generator.
    
    This class provides utilities to systematically evaluate different
    configurations of the phase generator and analyze their performance.
    """
    
    def __init__(self, d_model, phase_dim):
        """
        Initialize the parameter sweep.
        
        Args:
            d_model: Dimension of input embeddings
            phase_dim: Dimension of phase vectors
        """
        self.d_model = d_model
        self.phase_dim = phase_dim
        self.results = defaultdict(list)
        
    def _create_generator(self, hidden_dims, activation, phase_diversity, use_layer_norm):
        """
        Create a phase generator with the specified parameters.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            activation: Activation function type
            phase_diversity: Phase diversity parameter
            use_layer_norm: Whether to use layer normalization
            
        Returns:
            generator: Initialized EnhancedSemanticPhaseGenerator
        """
        return EnhancedSemanticPhaseGenerator(
            d_model=self.d_model,
            phase_dim=self.phase_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            phase_diversity=phase_diversity,
            use_layer_norm=use_layer_norm
        )
    
    def _evaluate_generator(self, generator, tokens, categories=None):
        """
        Evaluate a generator on the provided tokens.
        
        Args:
            generator: EnhancedSemanticPhaseGenerator to evaluate
            tokens: Tensor of shape [B, k, d_model] with input tokens
            categories: Optional tensor of shape [B, k] with category IDs
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Generate phase vectors
            phases = generator(tokens)
            
            # Calculate phase statistics
            phase_mean = phases.mean().item()
            phase_std = phases.std().item()
            phase_min = phases.min().item()
            phase_max = phases.max().item()
            
            # Calculate phase vector norms
            phase_norms = torch.norm(phases, dim=2)
            norm_mean = phase_norms.mean().item()
            norm_std = phase_norms.std().item()
            
            # Calculate similarity statistics
            analyzer = PhaseAnalyzer()
            similarity = analyzer.compute_similarity_matrix(phases)
            
            # Zero out diagonal (self-similarity)
            B, k, _ = similarity.shape
            for b in range(B):
                similarity[b].fill_diagonal_(0)
                
            sim_mean = similarity.mean().item()
            sim_std = similarity.std().item()
            
            # Calculate category-based metrics if categories are provided
            category_metrics = {}
            if categories is not None:
                category_metrics = analyzer.compute_category_similarities(phases, categories)
                intra_sim = category_metrics['intra_similarity'].mean().item()
                inter_sim = category_metrics['inter_similarity'].mean().item()
                contrast = category_metrics['contrast'].mean().item()
                category_metrics = {
                    'intra_similarity': intra_sim,
                    'inter_similarity': inter_sim,
                    'contrast': contrast
                }
            
            # Combine all metrics
            metrics = {
                'phase_mean': phase_mean,
                'phase_std': phase_std,
                'phase_min': phase_min,
                'phase_max': phase_max,
                'norm_mean': norm_mean,
                'norm_std': norm_std,
                'sim_mean': sim_mean,
                'sim_std': sim_std,
                **category_metrics
            }
            
            return metrics
    
    def run_sweep(self, tokens, categories=None, 
                 hidden_dims_list=None, 
                 activations=None, 
                 diversity_values=None,
                 layer_norm_values=None):
        """
        Run a parameter sweep over the specified configurations.
        
        Args:
            tokens: Tensor of shape [B, k, d_model] with input tokens
            categories: Optional tensor of shape [B, k] with category IDs
            hidden_dims_list: List of hidden dimension lists to try
            activations: List of activation functions to try
            diversity_values: List of diversity parameter values to try
            layer_norm_values: List of boolean values for layer norm
            
        Returns:
            results: Dictionary of evaluation results
        """
        # Default parameter values if not specified
        if hidden_dims_list is None:
            hidden_dims_list = [[64], [128, 64], [256, 128, 64]]
        
        if activations is None:
            activations = ['relu', 'gelu', 'leaky_relu', 'silu']
            
        if diversity_values is None:
            diversity_values = [0.5, 1.0, 2.0, 5.0]
            
        if layer_norm_values is None:
            layer_norm_values = [True, False]
        
        # Generate all parameter combinations
        parameter_combinations = list(itertools.product(
            hidden_dims_list, activations, diversity_values, layer_norm_values
        ))
        
        # Initialize results storage
        self.results = defaultdict(list)
        
        # Run evaluation for each parameter combination
        for i, (hidden_dims, activation, diversity, use_layer_norm) in enumerate(parameter_combinations):
            print(f"Evaluating configuration {i+1}/{len(parameter_combinations)}: "
                  f"hidden_dims={hidden_dims}, activation={activation}, "
                  f"diversity={diversity}, layer_norm={use_layer_norm}")
            
            # Create and evaluate generator
            generator = self._create_generator(
                hidden_dims=hidden_dims,
                activation=activation,
                phase_diversity=diversity,
                use_layer_norm=use_layer_norm
            )
            
            metrics = self._evaluate_generator(generator, tokens, categories)
            
            # Store configuration and metrics
            config = {
                'hidden_dims': hidden_dims,
                'activation': activation,
                'phase_diversity': diversity,
                'use_layer_norm': use_layer_norm
            }
            
            self.results['configurations'].append(config)
            self.results['metrics'].append(metrics)
        
        return self.results
    
    def rank_configurations(self, metric_name='contrast', ascending=False):
        """
        Rank configurations based on a specific metric.
        
        Args:
            metric_name: Name of the metric to rank by
            ascending: Whether to rank in ascending order
            
        Returns:
            ranked_configs: List of (configuration, metric) tuples
        """
        if not self.results:
            raise ValueError("No results available. Run a sweep first.")
        
        # Check if the metric exists in the results
        if metric_name not in self.results['metrics'][0]:
            raise ValueError(f"Metric '{metric_name}' not found in results.")
        
        # Extract configurations and corresponding metric values
        configs_with_metrics = []
        for config, metrics in zip(self.results['configurations'], self.results['metrics']):
            configs_with_metrics.append((config, metrics[metric_name]))
        
        # Sort by the specified metric
        ranked_configs = sorted(configs_with_metrics, key=lambda x: x[1], reverse=not ascending)
        
        return ranked_configs
    
    def plot_metric_distribution(self, metric_name='contrast', title=None):
        """
        Plot the distribution of a metric across all configurations.
        
        Args:
            metric_name: Name of the metric to plot
            title: Optional plot title
            
        Returns:
            fig: Matplotlib figure
        """
        if not self.results:
            raise ValueError("No results available. Run a sweep first.")
        
        # Check if the metric exists in the results
        if metric_name not in self.results['metrics'][0]:
            raise ValueError(f"Metric '{metric_name}' not found in results.")
        
        # Extract metric values
        metric_values = [metrics[metric_name] for metrics in self.results['metrics']]
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(metric_values, bins=20, alpha=0.7)
        
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Count')
        ax.set_title(title or f'Distribution of {metric_name}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_parameter_impact(self, param_name, metric_name='contrast'):
        """
        Plot the impact of a specific parameter on a metric.
        
        Args:
            param_name: Parameter to analyze ('hidden_dims', 'activation',
                        'phase_diversity', or 'use_layer_norm')
            metric_name: Metric to measure impact on
            
        Returns:
            fig: Matplotlib figure
        """
        if not self.results:
            raise ValueError("No results available. Run a sweep first.")
        
        # Check if the parameter exists in the configurations
        if param_name not in self.results['configurations'][0]:
            raise ValueError(f"Parameter '{param_name}' not found in configurations.")
        
        # Check if the metric exists in the results
        if metric_name not in self.results['metrics'][0]:
            raise ValueError(f"Metric '{metric_name}' not found in results.")
        
        # Extract unique parameter values
        param_values = []
        for config in self.results['configurations']:
            value = config[param_name]
            if isinstance(value, list):
                # For hidden_dims, use the length as a simple representation
                value = len(value)
            if value not in param_values:
                param_values.append(value)
        
        # Sort parameter values if they're numeric
        if all(isinstance(v, (int, float)) for v in param_values):
            param_values.sort()
        
        # Group metric values by parameter value
        grouped_metrics = defaultdict(list)
        for config, metrics in zip(self.results['configurations'], self.results['metrics']):
            param_value = config[param_name]
            if isinstance(param_value, list):
                param_value = len(param_value)
            grouped_metrics[param_value].append(metrics[metric_name])
        
        # Calculate statistics for each parameter value
        means = []
        stds = []
        for value in param_values:
            means.append(np.mean(grouped_metrics[value]))
            stds.append(np.std(grouped_metrics[value]))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot for categorical parameters, line plot for numeric
        if all(isinstance(v, (int, float)) for v in param_values):
            ax.errorbar(param_values, means, yerr=stds, marker='o', linestyle='-')
            ax.set_xlabel(param_name)
        else:
            x = np.arange(len(param_values))
            ax.bar(x, means, yerr=stds, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(param_values)
            ax.set_xlabel(param_name)
        
        ax.set_ylabel(metric_name)
        ax.set_title(f'Impact of {param_name} on {metric_name}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def save_results(self, filepath=None):
        """
        Save sweep results to a JSON file.
        
        Args:
            filepath: Path to save the results (default: timestamp-based filename)
            
        Returns:
            filepath: Path where results were saved
        """
        if not self.results:
            raise ValueError("No results available. Run a sweep first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"phase_generator_sweep_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = {
            'configurations': [],
            'metrics': self.results['metrics']
        }
        
        for config in self.results['configurations']:
            serializable_config = {
                'activation': config['activation'],
                'phase_diversity': config['phase_diversity'],
                'use_layer_norm': config['use_layer_norm'],
                'hidden_dims': config['hidden_dims']
            }
            serializable_results['configurations'].append(serializable_config)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def load_results(self, filepath):
        """
        Load sweep results from a JSON file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            results: Loaded results dictionary
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        print(f"Results loaded from {filepath}")
        return self.results

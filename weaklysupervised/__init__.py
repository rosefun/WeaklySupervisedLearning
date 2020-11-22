"""
The :mod:`.weakly supervised` module implements weakly-supervised learning
algorithms. These algorithms utilized noisy labeled data for classification tasks.
"""

from .BootstrappingSSL import BootstrappingNeuralNetworkClassifier

__all__ = ['BootstrappingNeuralNetworkClassifier']
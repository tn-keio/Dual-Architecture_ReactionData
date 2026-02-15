
# Abnormal Reaction Detection System
# Copyright (C) 2026
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class AnomalyDetector:
    def __init__(self, model):
        self.model = model

    def compress_reconstruct(self, X):
        """Reconstructs data using the autoencoder."""
        return self.model.predict(X)

    def calculate_mse(self, X):
        """Calculates Mean Squared Error for each sample."""
        reconstructions = self.compress_reconstruct(X)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1).flatten()
        return mse, reconstructions

    def calculate_elementwise_error(self, X, reconstructions=None):
        """Calculates squared error for each element (time point).
        
        Args:
            X: Input data (N, T, C).
            reconstructions: Pre-computed reconstructions. If None, will run predict.
        """
        if reconstructions is None:
            reconstructions = self.compress_reconstruct(X)
        # Result shape: (N, 38, 1) -> (N, 38)
        errors = np.power(X - reconstructions, 2).squeeze()
        return errors, reconstructions

    def determine_threshold(self, mse_scores, method='mean_std', std_devs=3.0, quantile=0.99):
        """
        Determines anomaly threshold based on distribution of MSE scores.
        
        Args:
            mse_scores: Array of MSE scores.
            method: 'mean_std', 'mad', or 'quantile'.
            std_devs: Number of standard deviations for 'mean_std' method.
            quantile: Quantile for 'quantile' method (e.g., 0.99 for top 1%).
            
        Returns:
            threshold: The calculated threshold value.
        """
        if method == 'mean_std':
            threshold = np.mean(mse_scores) + std_devs * np.std(mse_scores)
        elif method == 'mad':
            # Median + K * MAD. MAD = median(|x - median(x)|)
            # For normal distribution, sigma = 1.4826 * MAD.
            mean_val = np.median(mse_scores)
            mad = np.median(np.abs(mse_scores - mean_val))
            # 1.4826 is consistency constant for normal distribution
            threshold = mean_val + std_devs * 1.4826 * mad
        elif method == 'quantile':
            threshold = np.quantile(mse_scores, quantile)
        else:
            raise ValueError("Invalid method. Choose 'mean_std', 'mad', or 'quantile'.")
        
        return threshold

    def detect_anomalies(self, mse_scores, threshold):
        """Returns indices of anomalies based on threshold."""
        return np.where(mse_scores > threshold)[0]

    def find_unique_anomalies(self, X, candidate_indices, similarity_threshold=0.90, min_neighbors=3):
        """
        Filters candidates to find only unique anomalies (those without similar neighbors).
        
        Args:
            X: Original data array (N, T, C).
            candidate_indices: Indices of samples flagged as potential anomalies (e.g. high MSE).
            similarity_threshold: Cosine similarity threshold to consider two samples "similar".
                                  Higher means stricter similarity (must be almost identical).
            min_neighbors: Minimum number of similar samples (including itself) to consider it a "cluster".
                           If count < min_neighbors, it is an anomaly (unique).
                           If count >= min_neighbors, it is considered a repeating pattern (not anomaly).
            
        Returns:
            final_anomalies: Filtered list of indices (numpy array).
            details: Dictionary/DataFrame with details (optional extension).
        """
        if len(candidate_indices) == 0:
            return np.array([])

        # Flatten X for cosine similarity (N, Features)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Sub-select candidates
        X_candidates = X_flat[candidate_indices]
        
        # Compute cosine similarity between Candidates and ALL samples
        # shape: (Num_Candidates, Num_Total_Samples)
        # sim_matrix[i, j] is similarity between Candidate i and Sample j
        sim_matrix = cosine_similarity(X_candidates, X_flat)
        
        final_anomalies = []
        
        for i, idx in enumerate(candidate_indices):
            # Count samples with similarity > threshold
            # This count includes the sample itself (sim=1.0)
            similar_samples_count = np.sum(sim_matrix[i] >= similarity_threshold)
            
            # If the sample is isolated (few neighbors), it's a unique anomaly.
            # If it has many neighbors, it's a repeating pattern -> Ignore.
            if similar_samples_count < min_neighbors:
                final_anomalies.append(idx)
                
        return np.array(final_anomalies)

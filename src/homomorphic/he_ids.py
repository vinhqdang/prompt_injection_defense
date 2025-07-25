import tenseal as ts
import numpy as np
import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

class HomomorphicIDS:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]):
        """
        Initialize Homomorphic Encryption IDS
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
        """
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.model_weights = None
        self.encrypted_weights = None
        self.scale = 2**40
        
        # Initialize CKKS context
        self._setup_context(poly_modulus_degree, coeff_mod_bit_sizes)
        
    def _setup_context(self, poly_modulus_degree, coeff_mod_bit_sizes):
        """Setup TenSEAL CKKS context"""
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.generate_galois_keys()
        self.context.global_scale = self.scale
        
        print("Homomorphic encryption context initialized")
        print(f"Polynomial modulus degree: {poly_modulus_degree}")
        print(f"Coefficient modulus bit sizes: {coeff_mod_bit_sizes}")
    
    def train_linear_model(self, X_train, y_train, learning_rate=0.01, epochs=100):
        """
        Train a simple linear model for homomorphic evaluation
        This is a simplified logistic regression using gradient descent
        """
        print("Training linear model for homomorphic evaluation...")
        
        # Add bias term
        X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
        n_features = X_train_bias.shape[1]
        
        # Initialize weights
        weights = np.random.normal(0, 0.01, n_features)
        
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X_train_bias, weights)
            predictions = sigmoid(z)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(y_train * np.log(predictions + 1e-15) + 
                          (1 - y_train) * np.log(1 - predictions + 1e-15))
            
            # Backward pass
            gradient = np.dot(X_train_bias.T, (predictions - y_train)) / len(y_train)
            weights -= learning_rate * gradient
            
            if (epoch + 1) % 20 == 0:
                accuracy = accuracy_score(y_train, (predictions > 0.5).astype(int))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.model_weights = weights
        print(f"Linear model training completed. Final weights shape: {weights.shape}")
        return weights
    
    def encrypt_model(self):
        """Encrypt the trained model weights"""
        if self.model_weights is None:
            raise ValueError("No model weights found. Train the model first.")
        
        print("Encrypting model weights...")
        start_time = time.time()
        
        # Encrypt weights using CKKS
        self.encrypted_weights = ts.ckks_vector(self.context, self.model_weights.tolist())
        
        encryption_time = time.time() - start_time
        print(f"Model encryption completed in {encryption_time:.4f} seconds")
        
        return self.encrypted_weights
    
    def homomorphic_inference(self, X_test, batch_size=100):
        """
        Perform homomorphic inference on encrypted data
        """
        if self.encrypted_weights is None:
            raise ValueError("Model weights not encrypted. Call encrypt_model() first.")
        
        print("Performing homomorphic inference...")
        start_time = time.time()
        
        # Add bias term
        X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
        
        predictions = []
        n_batches = (len(X_test_bias) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_test_bias))
            batch = X_test_bias[start_idx:end_idx]
            
            batch_predictions = []
            for sample in batch:
                # Encrypt input data
                encrypted_input = ts.ckks_vector(self.context, sample.tolist())
                
                # Homomorphic dot product
                encrypted_result = encrypted_input.dot(self.encrypted_weights)
                
                # Decrypt result
                decrypted_result = encrypted_result.decrypt()[0]
                
                # Apply sigmoid approximation (simplified)
                # For homomorphic evaluation, we use a polynomial approximation
                # Here we use a simple threshold for binary classification
                prediction = 1 if decrypted_result > 0 else 0
                batch_predictions.append(prediction)
            
            predictions.extend(batch_predictions)
            
            if (i + 1) % 10 == 0:
                print(f"Processed batch {i+1}/{n_batches}")
        
        inference_time = time.time() - start_time
        print(f"Homomorphic inference completed in {inference_time:.4f} seconds")
        
        return np.array(predictions), inference_time
    
    def polynomial_approximation_inference(self, X_test, degree=3):
        """
        Alternative inference using polynomial approximation of sigmoid
        More suitable for homomorphic evaluation
        """
        if self.encrypted_weights is None:
            raise ValueError("Model weights not encrypted. Call encrypt_model() first.")
        
        print(f"Performing polynomial approximation inference (degree {degree})...")
        start_time = time.time()
        
        # Add bias term
        X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
        
        predictions = []
        
        for sample in X_test_bias:
            # Encrypt input
            encrypted_input = ts.ckks_vector(self.context, sample.tolist())
            
            # Homomorphic dot product
            encrypted_z = encrypted_input.dot(self.encrypted_weights)
            
            # Polynomial approximation of sigmoid: f(x) ≈ 0.5 + 0.25*x - 0.125*x^3
            # This is a common approximation for x in [-2, 2]
            encrypted_z_squared = encrypted_z * encrypted_z
            encrypted_z_cubed = encrypted_z_squared * encrypted_z
            
            # Compute polynomial approximation
            encrypted_sigmoid = (encrypted_z * 0.25) - (encrypted_z_cubed * 0.125)
            
            # Add constant term 0.5
            constant_vector = ts.ckks_vector(self.context, [0.5])
            encrypted_result = encrypted_sigmoid + constant_vector
            
            # Decrypt and classify
            decrypted_result = encrypted_result.decrypt()[0]
            prediction = 1 if decrypted_result > 0.5 else 0
            predictions.append(prediction)
        
        inference_time = time.time() - start_time
        print(f"Polynomial approximation inference completed in {inference_time:.4f} seconds")
        
        return np.array(predictions), inference_time
    
    def evaluate_homomorphic_model(self, X_test, y_test, method='simple'):
        """Evaluate the homomorphic model"""
        if method == 'simple':
            y_pred, inference_time = self.homomorphic_inference(X_test)
        elif method == 'polynomial':
            y_pred, inference_time = self.polynomial_approximation_inference(X_test)
        else:
            raise ValueError("Method must be 'simple' or 'polynomial'")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            'method': f'homomorphic_{method}',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'predictions': y_pred
        }
        
        print(f"\nHomomorphic {method.title()} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Inference Time: {inference_time:.4f} seconds")
        
        return results
    
    def save_context(self, filepath='models/saved/he_context.bin'):
        """Save the homomorphic encryption context"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        context_data = {
            'context': self.context.serialize(),
            'weights': self.model_weights.tolist() if self.model_weights is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(context_data, f)
        
        print(f"HE context saved to {filepath}")
    
    def load_context(self, filepath='models/saved/he_context.bin'):
        """Load the homomorphic encryption context"""
        with open(filepath, 'rb') as f:
            context_data = pickle.load(f)
        
        # Deserialize context
        self.context = ts.context_from(context_data['context'])
        
        if context_data['weights'] is not None:
            self.model_weights = np.array(context_data['weights'])
            # Re-encrypt weights
            self.encrypt_model()
        
        print(f"HE context loaded from {filepath}")
    
    def compare_with_plaintext(self, X_test, y_test):
        """Compare homomorphic results with plaintext computation"""
        if self.model_weights is None:
            raise ValueError("No model weights found.")
        
        print("Comparing homomorphic vs plaintext computation...")
        
        # Plaintext inference
        X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
        z_plaintext = np.dot(X_test_bias, self.model_weights)
        
        # Simple threshold for binary classification
        y_pred_plaintext = (z_plaintext > 0).astype(int)
        
        # Homomorphic inference
        y_pred_homomorphic, _ = self.homomorphic_inference(X_test)
        
        # Compare accuracy
        acc_plaintext = accuracy_score(y_test, y_pred_plaintext)
        acc_homomorphic = accuracy_score(y_test, y_pred_homomorphic)
        
        print(f"Plaintext Accuracy: {acc_plaintext:.4f}")
        print(f"Homomorphic Accuracy: {acc_homomorphic:.4f}")
        print(f"Accuracy Difference: {abs(acc_plaintext - acc_homomorphic):.4f}")
        
        return {
            'plaintext_accuracy': acc_plaintext,
            'homomorphic_accuracy': acc_homomorphic,
            'accuracy_difference': abs(acc_plaintext - acc_homomorphic)
        }
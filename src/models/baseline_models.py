import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import joblib
import os

class BaselineModels:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        """Train Random Forest model"""
        print("Training Random Forest...")
        start_time = time.time()
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.models['random_forest'] = rf
        
        print(f"Random Forest training completed in {training_time:.2f} seconds")
        return rf
    
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None):
        """Train XGBoost model"""
        print("Training XGBoost...")
        start_time = time.time()
        
        # Convert to DMatrix for better performance
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'
        }
        
        evals = []
        if X_test is not None and y_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test)
            evals = [(dtrain, 'train'), (dtest, 'eval')]
        
        xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        training_time = time.time() - start_time
        self.models['xgboost'] = xgb_model
        
        print(f"XGBoost training completed in {training_time:.2f} seconds")
        return xgb_model
    
    def train_deep_learning(self, X_train, y_train, X_test=None, y_test=None, 
                          epochs=50, batch_size=512, learning_rate=0.001):
        """Train Deep Learning model"""
        print("Training Deep Learning model...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define neural network
        class DeepIDSModel(nn.Module):
            def __init__(self, input_size):
                super(DeepIDSModel, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = DeepIDSModel(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        start_time = time.time()
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        self.models['deep_learning'] = model
        
        print(f"Deep Learning training completed in {training_time:.2f} seconds")
        return model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        start_time = time.time()
        
        if model_name == 'random_forest':
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        elif model_name == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            y_prob = model.predict(dtest)
            y_pred = (y_prob > 0.5).astype(int)
            
        elif model_name == 'deep_learning':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_prob = model(X_test_tensor).cpu().numpy().squeeze()
                y_pred = (y_prob > 0.5).astype(int)
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Inference Time: {inference_time:.4f} seconds")
        
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        for model_name in self.models.keys():
            self.evaluate_model(model_name, X_test, y_test)
    
    def save_models(self, save_dir='models/saved'):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'random_forest':
                joblib.dump(model, f'{save_dir}/random_forest.pkl')
            elif name == 'xgboost':
                model.save_model(f'{save_dir}/xgboost.json')
            elif name == 'deep_learning':
                torch.save(model.state_dict(), f'{save_dir}/deep_learning.pth')
        
        print(f"Models saved to {save_dir}")
    
    def get_comparison_table(self):
        """Get comparison table of all models"""
        if not self.results:
            print("No results available. Please evaluate models first.")
            return None
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'Inference Time (s)': f"{results['inference_time']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        return df
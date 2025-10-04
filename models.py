"""
Optimized ML Models for Exoplanet Detection
Balanced for accuracy and speed
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os
import time
from typing import Dict, Tuple, Any

class ExoplanetModelEnsemble:
    def __init__(self, input_shape=(256, 1), n_features=20):
        """
        Initialize ensemble of models optimized for speed-accuracy balance
        """
        self.input_shape = input_shape
        self.n_features = n_features
        self.models = {}
        self.model_weights = {}
        self.training_history = {}
        
    def create_lightweight_cnn(self) -> keras.Model:
        """
        Create a lightweight CNN optimized for speed
        Uses depthwise separable convolutions and efficient architecture
        """
        model = models.Sequential([
            # First block - feature extraction
            layers.Conv1D(32, 7, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Second block - depthwise separable for efficiency
            layers.SeparableConv1D(64, 5, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            # Third block
            layers.SeparableConv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Global pooling instead of flatten for efficiency
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Use efficient optimizer
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_efficient_lstm(self) -> keras.Model:
        """
        Create an efficient LSTM model for temporal patterns
        Uses bidirectional LSTM with attention mechanism
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # LSTM layers with return sequences for attention
        lstm_out = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(inputs)
        
        # Attention mechanism (simplified)
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)  # 128 = 64*2 (bidirectional)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        attended = layers.Multiply()([lstm_out, attention])
        attended = layers.GlobalAveragePooling1D()(attended)
        
        # Dense layers
        dense = layers.Dense(64, activation='relu')(attended)
        dense = layers.Dropout(0.4)(dense)
        dense = layers.Dense(32, activation='relu')(dense)
        dense = layers.Dropout(0.3)(dense)
        outputs = layers.Dense(1, activation='sigmoid')(dense)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_hybrid_model(self) -> keras.Model:
        """
        Create a hybrid model combining CNN and classical features
        Optimized for both accuracy and interpretability
        """
        # CNN branch for sequences
        sequence_input = layers.Input(shape=self.input_shape, name='sequences')
        
        # Lightweight CNN
        conv1 = layers.Conv1D(32, 7, activation='relu')(sequence_input)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.MaxPooling1D(2)(conv1)
        
        conv2 = layers.SeparableConv1D(64, 5, activation='relu')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.MaxPooling1D(2)(conv2)
        
        conv3 = layers.SeparableConv1D(128, 3, activation='relu')(conv2)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.GlobalAveragePooling1D()(conv3)
        
        # Classical features branch
        features_input = layers.Input(shape=(self.n_features,), name='features')
        features_dense = layers.Dense(32, activation='relu')(features_input)
        features_dense = layers.BatchNormalization()(features_dense)
        features_dense = layers.Dropout(0.3)(features_dense)
        
        # Combine branches
        combined = layers.Concatenate()([conv3, features_dense])
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.4)(combined)
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        outputs = layers.Dense(1, activation='sigmoid')(combined)
        
        model = models.Model(
            inputs=[sequence_input, features_input], 
            outputs=outputs
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_classical_models(self):
        """Create classical ML models for comparison and ensemble"""
        models_dict = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        return models_dict
    
    def train_deep_models(self, data_splits: Dict, epochs: int = 50, batch_size: int = 32):
        """Train deep learning models"""
        print("Training deep learning models...")
        
        # Callbacks for efficient training
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train CNN
        print("Training CNN...")
        start_time = time.time()
        cnn_model = self.create_lightweight_cnn()
        
        cnn_history = cnn_model.fit(
            data_splits['train']['sequences'],
            data_splits['train']['labels'],
            validation_data=(data_splits['val']['sequences'], data_splits['val']['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        cnn_time = time.time() - start_time
        self.models['cnn'] = cnn_model
        self.training_history['cnn'] = cnn_history.history
        
        # Train LSTM
        print("Training LSTM...")
        start_time = time.time()
        lstm_model = self.create_efficient_lstm()
        
        lstm_history = lstm_model.fit(
            data_splits['train']['sequences'],
            data_splits['train']['labels'],
            validation_data=(data_splits['val']['sequences'], data_splits['val']['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        lstm_time = time.time() - start_time
        self.models['lstm'] = lstm_model
        self.training_history['lstm'] = lstm_history.history
        
        # Train Hybrid Model
        print("Training Hybrid Model...")
        start_time = time.time()
        hybrid_model = self.create_hybrid_model()
        
        hybrid_history = hybrid_model.fit(
            [data_splits['train']['sequences'], data_splits['train']['features']],
            data_splits['train']['labels'],
            validation_data=(
                [data_splits['val']['sequences'], data_splits['val']['features']],
                data_splits['val']['labels']
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        hybrid_time = time.time() - start_time
        self.models['hybrid'] = hybrid_model
        self.training_history['hybrid'] = hybrid_history.history
        
        print(f"Training times - CNN: {cnn_time:.1f}s, LSTM: {lstm_time:.1f}s, Hybrid: {hybrid_time:.1f}s")
    
    def train_classical_models(self, data_splits: Dict):
        """Train classical ML models"""
        print("Training classical ML models...")
        
        classical_models = self.create_classical_models()
        
        X_train = data_splits['train']['features']
        y_train = data_splits['train']['labels']
        
        for name, model in classical_models.items():
            print(f"Training {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            self.models[name] = model
            
            print(f"{name} training time: {training_time:.2f}s")
    
    def evaluate_models(self, data_splits: Dict) -> Dict:
        """Evaluate all models and return performance metrics"""
        print("Evaluating models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Prepare test data based on model type
            if model_name in ['cnn', 'lstm']:
                X_test = data_splits['test']['sequences']
            elif model_name == 'hybrid':
                X_test = [data_splits['test']['sequences'], data_splits['test']['features']]
            else:  # Classical models
                X_test = data_splits['test']['features']
            
            y_test = data_splits['test']['labels']
            
            # Make predictions
            start_time = time.time()
            if model_name in ['cnn', 'lstm', 'hybrid']:
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            
            inference_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[model_name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score'],
                'inference_time': inference_time,
                'predictions': y_pred_proba
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Inference time: {inference_time:.3f}s")
        
        return results
    
    def create_ensemble_prediction(self, results: Dict, data_splits: Dict) -> np.ndarray:
        """Create ensemble prediction using weighted voting"""
        print("Creating ensemble prediction...")
        
        # Calculate weights based on AUC scores
        auc_scores = {name: result['auc'] for name, result in results.items()}
        total_auc = sum(auc_scores.values())
        weights = {name: auc / total_auc for name, auc in auc_scores.items()}
        
        self.model_weights = weights
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(data_splits['test']['labels']))
        
        for model_name, weight in weights.items():
            ensemble_pred += weight * results[model_name]['predictions']
        
        # Calculate ensemble metrics
        y_test = data_splits['test']['labels']
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        
        ensemble_accuracy = np.mean(ensemble_binary == y_test)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'auc': ensemble_auc,
            'predictions': ensemble_pred,
            'weights': weights
        }
        
        print(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, AUC: {ensemble_auc:.4f}")
        print(f"Model weights: {weights}")
        
        return ensemble_pred
    
    def save_models(self, output_dir: str = 'models'):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name in ['cnn', 'lstm', 'hybrid']:
                model.save(f'{output_dir}/{model_name}_model.h5')
            else:
                joblib.dump(model, f'{output_dir}/{model_name}_model.pkl')
        
        # Save model weights and training history
        joblib.dump(self.model_weights, f'{output_dir}/ensemble_weights.pkl')
        joblib.dump(self.training_history, f'{output_dir}/training_history.pkl')
        
        print(f"Models saved to {output_dir}")
    
    def load_models(self, model_dir: str = 'models'):
        """Load pre-trained models"""
        print("Loading pre-trained models...")
        
        # Load deep learning models
        for model_name in ['cnn', 'lstm', 'hybrid']:
            model_path = f'{model_dir}/{model_name}_model.h5'
            if os.path.exists(model_path):
                self.models[model_name] = keras.models.load_model(model_path)
        
        # Load classical models
        for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']:
            model_path = f'{model_dir}/{model_name}_model.pkl'
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
        
        # Load ensemble weights
        weights_path = f'{model_dir}/ensemble_weights.pkl'
        if os.path.exists(weights_path):
            self.model_weights = joblib.load(weights_path)
        
        print(f"Loaded {len(self.models)} models")

if __name__ == "__main__":
    # Load processed data
    data_splits = {}
    for split in ['train', 'val', 'test']:
        data = np.load(f'data/processed/{split}_data.npz')
        data_splits[split] = {
            'sequences': data['sequences'],
            'features': data['features'],
            'labels': data['labels']
        }
    
    # Initialize model ensemble
    n_features = data_splits['train']['features'].shape[1]
    ensemble = ExoplanetModelEnsemble(n_features=n_features)
    
    # Train models
    ensemble.train_deep_models(data_splits, epochs=30)
    ensemble.train_classical_models(data_splits)
    
    # Evaluate models
    results = ensemble.evaluate_models(data_splits)
    
    # Create ensemble
    ensemble_pred = ensemble.create_ensemble_prediction(results, data_splits)
    
    # Save models
    ensemble.save_models()
    
    print("Model training and evaluation complete!")

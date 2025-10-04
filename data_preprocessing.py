"""
Data Preprocessing Pipeline for Exoplanet Detection
Optimized for speed and accuracy balance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.stats import skew, kurtosis
import lightkurve as lk
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ExoplanetPreprocessor:
    def __init__(self, target_length=2048):
        """
        Initialize preprocessor with target length for uniform data
        2048 points provides good balance between detail and speed
        """
        self.target_length = target_length
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_light_curve(self, filepath):
        """Load light curve from various formats"""
        try:
            if filepath.endswith('.fits'):
                lc = lk.read(filepath)
                return lc.time.value, lc.flux.value
            elif filepath.endswith('.npy'):
                data = np.load(filepath, allow_pickle=True).item()
                return data['time'], data['flux']
            else:
                # Assume CSV format
                df = pd.read_csv(filepath)
                return df['time'].values, df['flux'].values
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def normalize_flux(self, flux):
        """Normalize flux to remove stellar brightness variations"""
        # Remove NaN values
        flux = flux[~np.isnan(flux)]
        
        # Normalize to median
        flux_normalized = flux / np.median(flux)
        
        # Remove outliers (beyond 3 sigma)
        sigma = np.std(flux_normalized)
        mask = np.abs(flux_normalized - 1.0) < 3 * sigma
        
        return flux_normalized[mask]
    
    def detrend_light_curve(self, time, flux, window_length=101):
        """Remove long-term trends using Savitzky-Golay filter"""
        if len(flux) < window_length:
            window_length = len(flux) // 2
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < 3:
                return flux
        
        try:
            # Apply Savitzky-Golay filter to remove trends
            trend = signal.savgol_filter(flux, window_length, 3)
            detrended = flux / trend
            return detrended
        except:
            return flux
    
    def resample_light_curve(self, time, flux, target_length=None):
        """Resample light curve to uniform length"""
        if target_length is None:
            target_length = self.target_length
            
        if len(flux) == target_length:
            return time, flux
        
        # Create uniform time grid
        time_uniform = np.linspace(time.min(), time.max(), target_length)
        
        # Interpolate flux to uniform grid
        flux_uniform = np.interp(time_uniform, time, flux)
        
        return time_uniform, flux_uniform
    
    def extract_features(self, flux):
        """Extract statistical and frequency domain features for classical ML"""
        features = {}
        
        # Statistical features
        features['mean'] = np.mean(flux)
        features['std'] = np.std(flux)
        features['var'] = np.var(flux)
        features['skewness'] = skew(flux)
        features['kurtosis'] = kurtosis(flux)
        features['min'] = np.min(flux)
        features['max'] = np.max(flux)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(flux)
        features['q25'] = np.percentile(flux, 25)
        features['q75'] = np.percentile(flux, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Frequency domain features
        try:
            fft = np.fft.fft(flux)
            power_spectrum = np.abs(fft)**2
            freqs = np.fft.fftfreq(len(flux))
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            features['dominant_frequency'] = freqs[dominant_freq_idx]
            features['dominant_power'] = power_spectrum[dominant_freq_idx]
            
            # Spectral centroid
            features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2])
            
            # Spectral rolloff
            cumsum_power = np.cumsum(power_spectrum[:len(power_spectrum)//2])
            rolloff_idx = np.where(cumsum_power >= 0.85 * cumsum_power[-1])[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = freqs[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = 0
                
        except:
            # If FFT fails, set default values
            features['dominant_frequency'] = 0
            features['dominant_power'] = 0
            features['spectral_centroid'] = 0
            features['spectral_rolloff'] = 0
        
        # Transit-specific features
        # Look for periodic dips
        flux_smooth = signal.medfilt(flux, kernel_size=5)
        dips = flux_smooth < (np.median(flux_smooth) - 2 * np.std(flux_smooth))
        features['n_dips'] = np.sum(dips)
        features['dip_fraction'] = np.mean(dips)
        
        # Autocorrelation features
        try:
            autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find first minimum (potential period)
            if len(autocorr) > 10:
                features['autocorr_period'] = np.argmin(autocorr[1:10]) + 1
                features['autocorr_strength'] = autocorr[features['autocorr_period']]
            else:
                features['autocorr_period'] = 0
                features['autocorr_strength'] = 0
        except:
            features['autocorr_period'] = 0
            features['autocorr_strength'] = 0
        
        return features
    
    def create_sequences(self, flux, sequence_length=256, overlap=0.5):
        """Create overlapping sequences for LSTM/CNN training"""
        step = int(sequence_length * (1 - overlap))
        sequences = []
        
        for i in range(0, len(flux) - sequence_length + 1, step):
            sequences.append(flux[i:i + sequence_length])
        
        return np.array(sequences)
    
    def process_dataset(self, metadata_df, data_dir='data'):
        """Process entire dataset for training"""
        print("Processing dataset for training...")
        
        processed_data = {
            'flux_sequences': [],
            'features': [],
            'labels': [],
            'filenames': []
        }
        
        failed_files = []
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing files"):
            try:
                # Load light curve
                if 'synthetic' in row['filename']:
                    time, flux = self.load_light_curve(row['filename'])
                else:
                    filepath = os.path.join(data_dir, row['filename'])
                    time, flux = self.load_light_curve(filepath)
                
                if time is None or flux is None:
                    failed_files.append(row['filename'])
                    continue
                
                # Preprocess
                flux_norm = self.normalize_flux(flux)
                if len(flux_norm) < 100:  # Skip very short light curves
                    failed_files.append(row['filename'])
                    continue
                
                flux_detrended = self.detrend_light_curve(time[:len(flux_norm)], flux_norm)
                time_uniform, flux_uniform = self.resample_light_curve(
                    time[:len(flux_detrended)], flux_detrended
                )
                
                # Extract features
                features = self.extract_features(flux_uniform)
                
                # Create sequences for deep learning
                sequences = self.create_sequences(flux_uniform, sequence_length=256)
                
                # Store processed data
                for seq in sequences:
                    processed_data['flux_sequences'].append(seq)
                    processed_data['features'].append(list(features.values()))
                    processed_data['labels'].append(int(row['has_exoplanet']))
                    processed_data['filenames'].append(row['filename'])
                
            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")
                failed_files.append(row['filename'])
                continue
        
        print(f"Successfully processed {len(processed_data['labels'])} sequences")
        print(f"Failed to process {len(failed_files)} files")
        
        # Convert to numpy arrays
        processed_data['flux_sequences'] = np.array(processed_data['flux_sequences'])
        processed_data['features'] = np.array(processed_data['features'])
        processed_data['labels'] = np.array(processed_data['labels'])
        
        # Scale features
        processed_data['features'] = self.feature_scaler.fit_transform(processed_data['features'])
        
        return processed_data
    
    def prepare_training_data(self, processed_data, test_size=0.2, val_size=0.1):
        """Split data into training, validation, and test sets"""
        print("Preparing training data splits...")
        
        X_sequences = processed_data['flux_sequences']
        X_features = processed_data['features']
        y = processed_data['labels']
        
        # First split: separate test set
        X_seq_temp, X_seq_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
            X_sequences, X_features, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_seq_train, X_seq_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
            X_seq_temp, X_feat_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Reshape sequences for CNN input (add channel dimension)
        X_seq_train = X_seq_train.reshape(-1, X_seq_train.shape[1], 1)
        X_seq_val = X_seq_val.reshape(-1, X_seq_val.shape[1], 1)
        X_seq_test = X_seq_test.reshape(-1, X_seq_test.shape[1], 1)
        
        data_splits = {
            'train': {
                'sequences': X_seq_train,
                'features': X_feat_train,
                'labels': y_train
            },
            'val': {
                'sequences': X_seq_val,
                'features': X_feat_val,
                'labels': y_val
            },
            'test': {
                'sequences': X_seq_test,
                'features': X_feat_test,
                'labels': y_test
            }
        }
        
        print(f"Training set: {len(y_train)} samples")
        print(f"Validation set: {len(y_val)} samples")
        print(f"Test set: {len(y_test)} samples")
        print(f"Positive class ratio - Train: {np.mean(y_train):.3f}, Val: {np.mean(y_val):.3f}, Test: {np.mean(y_test):.3f}")
        
        return data_splits
    
    def save_processed_data(self, processed_data, data_splits, output_dir='data/processed'):
        """Save processed data for later use"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full processed dataset
        np.savez_compressed(
            f'{output_dir}/processed_dataset.npz',
            flux_sequences=processed_data['flux_sequences'],
            features=processed_data['features'],
            labels=processed_data['labels']
        )
        
        # Save data splits
        for split_name, split_data in data_splits.items():
            np.savez_compressed(
                f'{output_dir}/{split_name}_data.npz',
                sequences=split_data['sequences'],
                features=split_data['features'],
                labels=split_data['labels']
            )
        
        print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ExoplanetPreprocessor(target_length=2048)
    
    # Load metadata
    metadata_df = pd.read_csv('data/training_metadata.csv')
    
    # Process dataset
    processed_data = preprocessor.process_dataset(metadata_df)
    
    # Prepare training splits
    data_splits = preprocessor.prepare_training_data(processed_data)
    
    # Save processed data
    preprocessor.save_processed_data(processed_data, data_splits)
    
    print("Data preprocessing complete!")

import pandas as pd
import numpy as np
import librosa
import io
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

def extract_audio_from_bytes(audio_bytes):
    """Extract audio array from bytes data."""
    try:
        audio_buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_buffer)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def extract_audio_features(audio_data, sample_rate):
    """Extract meaningful features from audio data."""
    features = {}
    
    # Basic audio statistics
    features['duration'] = len(audio_data) / sample_rate
    features['sample_rate'] = sample_rate
    features['mean_amplitude'] = np.mean(audio_data)
    features['std_amplitude'] = np.std(audio_data)
    features['max_amplitude'] = np.max(audio_data)
    features['min_amplitude'] = np.min(audio_data)
    features['rms'] = np.sqrt(np.mean(audio_data**2))
    
    # Zero crossing rate
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    
    # Spectral features
    try:
        # MFCCs (Mel-frequency cepstral coefficients) - most important for speech
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
    except Exception as e:
        print(f"Error extracting spectral features: {e}")
    
    return features

def process_dataset(df, dataset_name):
    """Process entire dataset and extract features."""
    print(f"Processing {dataset_name} dataset...")
    features_list = []
    labels_list = []
    
    for idx in range(len(df)):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(df)}...")
        
        audio_dict = df.iloc[idx]['audio']
        label = df.iloc[idx]['label']
        
        if 'bytes' in audio_dict:
            audio_data, sample_rate = extract_audio_from_bytes(audio_dict['bytes'])
            if audio_data is not None:
                features = extract_audio_features(audio_data, sample_rate)
                features_list.append(features)
                labels_list.append(label)
    
    return features_list, labels_list

# Load the data
print("Loading data...")
train_df = pd.read_parquet('train-00000-of-00001.parquet')
test_df = pd.read_parquet('test-00000-of-00001.parquet')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Process training data
train_features, train_labels = process_dataset(train_df, "training")
print(f"Successfully processed {len(train_features)} training samples")

# Process test data
test_features, test_labels = process_dataset(test_df, "test")
print(f"Successfully processed {len(test_features)} test samples")

# Convert to DataFrame for easier handling
train_features_df = pd.DataFrame(train_features)
test_features_df = pd.DataFrame(test_features)

print(f"Training features shape: {train_features_df.shape}")
print(f"Test features shape: {test_features_df.shape}")

# Fill any NaN values with median
train_features_df = train_features_df.fillna(train_features_df.median())
test_features_df = test_features_df.fillna(train_features_df.median())

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_features_df)
X_test_scaled = scaler.transform(test_features_df)

# Train the model
print("Training Random Forest model...")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, train_labels)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Make predictions
print("Making predictions...")
start_time = time.time()
train_predictions = rf_model.predict(X_train_scaled)
train_pred_time = time.time() - start_time

start_time = time.time()
test_predictions = rf_model.predict(X_test_scaled)
test_pred_time = time.time() - start_time

# Calculate accuracies
train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"\n=== Model Performance ===")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Average prediction time per sample: {test_pred_time/len(test_predictions)*1000:.2f} ms")

# Detailed classification report
print(f"\n=== Detailed Classification Report ===")
print(classification_report(test_labels, test_predictions))

# Confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - Digit Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': train_features_df.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Audio Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the trained model and scaler
print("Saving model and scaler...")
with open('digit_classifier_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(train_features_df.columns), f)

print("Model training and evaluation completed!")
print(f"Model saved as 'digit_classifier_model.pkl'")
print(f"Scaler saved as 'feature_scaler.pkl'")
print(f"Feature names saved as 'feature_names.pkl'")

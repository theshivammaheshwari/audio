
import librosa
import numpy as np
import joblib

class SyntheticSpeechDetector:
    def __init__(self, model_path, scaler_path):
        """Initialize detector with trained model"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # ⚠️ VERIFY MODEL CLASSES
        print(f"Model loaded - classes: {self.model.classes_}")
        assert list(self.model.classes_) == [0, 1], "Model classes must be [0, 1]!"
        
    def extract_features(self, audio_path, sr=16000):
        """Extract same features as training"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, duration=4.0)
            
            # Preprocessing
            y_trim, _ = librosa.effects.trim(y, top_db=20)
            
            # Pad or truncate
            target_length = 4 * sr
            if len(y_trim) < target_length:
                y_trim = np.pad(y_trim, (0, target_length - len(y_trim)))
            else:
                y_trim = y_trim[:target_length]
            
            # Normalize
            y_trim = y_trim / (np.max(np.abs(y_trim)) + 1e-6)
            
            features = []
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=20)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            
            # Spectral
            spectral_centroids = librosa.feature.spectral_centroid(y=y_trim, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trim, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)
            
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y_trim)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y_trim, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            
            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y_trim, sr=sr)
            features.extend(np.mean(contrast, axis=1))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def detect(self, audio_path):
        """Detect if audio is bonafide or spoof"""
        try:
            # Extract features
            features = self.extract_features(audio_path)
            
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # ✅ CORRECT INTERPRETATION
            # model.classes_ = [0, 1]
            # probabilities[0] = probability of class 0 (bonafide)
            # probabilities[1] = probability of class 1 (spoof)
            
            bonafide_prob = probabilities[0]
            spoof_prob = probabilities[1]
            
            result = {
                'prediction': 'Bonafide' if prediction == 0 else 'Synthetic',
                'confidence': max(bonafide_prob, spoof_prob),
                'bonafide_probability': float(bonafide_prob),
                'synthetic_probability': float(spoof_prob)
            }
            
            return result
            
        except Exception as e:
            return {'error': str(e)}

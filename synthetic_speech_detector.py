import librosa
import numpy as np
import joblib

class SyntheticSpeechDetector:
    def __init__(self, model_path, scaler_path):
        """Initialize detector with trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        # Verify model classes
        if not np.array_equal(self.model.classes_, [0, 1]):
            raise ValueError(f"Invalid model classes: {self.model.classes_}")

        print(f"✓ Detector loaded - Model classes: {self.model.classes_}")

    def extract_features(self, audio_path, sr=16000, duration=4.0):
        """Extract 67-dimensional feature vector"""
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            y_trim, _ = librosa.effects.trim(y, top_db=20)

            # Check minimum length
            if len(y_trim) < sr * 0.5:
                return None

            # Fixed length
            target_length = int(duration * sr)
            if len(y_trim) < target_length:
                y_trim = np.pad(y_trim, (0, target_length - len(y_trim)))
            else:
                y_trim = y_trim[:target_length]

            # Normalize
            max_amp = np.max(np.abs(y_trim))
            if max_amp > 0:
                y_trim = y_trim / max_amp

            features = []

            # MFCC (40)
            mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=20, n_fft=512, hop_length=256)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))

            # Spectral (6)
            features.append(np.mean(librosa.feature.spectral_centroid(y=y_trim, sr=sr)))
            features.append(np.std(librosa.feature.spectral_centroid(y=y_trim, sr=sr)))
            features.append(np.mean(librosa.feature.spectral_rolloff(y=y_trim, sr=sr)))
            features.append(np.std(librosa.feature.spectral_rolloff(y=y_trim, sr=sr)))
            features.append(np.mean(librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)))
            features.append(np.std(librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)))

            # ZCR (2)
            zcr = librosa.feature.zero_crossing_rate(y_trim)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))

            # Chroma (12)
            chroma = librosa.feature.chroma_stft(y=y_trim, sr=sr)
            features.extend(np.mean(chroma, axis=1))

            # Spectral Contrast (7)
            contrast = librosa.feature.spectral_contrast(y=y_trim, sr=sr)
            features.extend(np.mean(contrast, axis=1))

            return np.array(features).reshape(1, -1)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def detect(self, audio_path):
        """
        Detect if audio is real (bonafide) or fake (synthetic)

        Returns:
            dict: {
                'prediction': 'Bonafide' or 'Synthetic',
                'confidence': float,
                'bonafide_probability': float,
                'synthetic_probability': float
            }
        """
        try:
            # Extract features
            features = self.extract_features(audio_path)

            if features is None:
                return {
                    'error': 'Failed to extract features. Audio may be too short or corrupted.'
                }

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]

            # probabilities[0] = bonafide (class 0)
            # probabilities[1] = spoof (class 1)

            return {
                'prediction': 'Bonafide' if prediction == 0 else 'Synthetic',
                'confidence': float(max(probabilities)),
                'bonafide_probability': float(probabilities[0]),
                'synthetic_probability': float(probabilities[1])
            }

        except Exception as e:
            return {'error': f'Detection error: {str(e)}'}

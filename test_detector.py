#!/usr/bin/env python3
import sys
import os
from synthetic_speech_detector import SyntheticSpeechDetector

def main():
    if len(sys.argv) != 2:
        print('Usage: python test_detector.py <audio_file>')
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f'Error: File {audio_file} not found')
        sys.exit(1)

    # Initialize detector
    try:
        detector = SyntheticSpeechDetector('xgboost_model.pkl', 'scaler.pkl')
        print('✓ Model loaded successfully')
    except Exception as e:
        print(f'Error loading model: {e}')
        sys.exit(1)

    # Detect
    print(f'Processing: {audio_file}')
    result = detector.detect(audio_file)

    if 'error' in result:
        print(f'Error: {result[error]}')
    else:
        print('\nResults:')
        print(f'Prediction: {result[prediction]}')
        print(f'Confidence: {result[confidence]:.2%}')
        print(f'Bonafide probability: {result[bonafide_probability]:.4f}')
        print(f'Synthetic probability: {result[synthetic_probability]:.4f}')

if __name__ == '__main__':
    main()

import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
from synthetic_speech_detector import SyntheticSpeechDetector

# Page config
st.set_page_config(
    page_title="Voice Detection System",
    page_icon="🎤",
    layout="wide"
)

# Title
st.title("🎤 Synthetic Speech Detection System")
st.markdown("### Detect if voice is Real (Human) or Fake (AI-Generated)")

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
**Model Performance:**
- Accuracy: 91.4%
- Equal Error Rate: 8.2%
- Training Samples: 3,000+
- Features: MFCC, Spectral, Chroma
""")

st.sidebar.header("📝 Instructions")
st.sidebar.markdown("""
1. Upload an audio file (WAV, MP3, FLAC)
2. Click 'Analyze Voice'
3. Get instant results!

**Supported formats:**
- WAV, MP3, FLAC
- Any duration (auto-processed)
- Recommended: 16kHz sampling rate
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎵 Upload Audio File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac'],
        help="Upload WAV, MP3, or FLAC audio files"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size / 1024:.1f} KB")
        
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Analyze button
        if st.button("🔍 Analyze Voice", type="primary"):
            with st.spinner("🔄 Analyzing audio... Please wait"):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load detector
                    @st.cache_resource
                    def load_detector():
                        return SyntheticSpeechDetector('xgboost_model.pkl', 'scaler.pkl')
                    
                    detector = load_detector()
                    
                    # Detect
                    result = detector.detect(tmp_path)
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    # Display results
                    if 'error' in result:
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        # Results section
                        st.header("📊 Analysis Results")
                        
                        # Main result
                        if result['prediction'] == 'Bonafide':
                            st.success("✅ **REAL VOICE** - Human Speech Detected")
                            st.balloons()
                        else:
                            st.error("❌ **FAKE VOICE** - AI-Generated Speech Detected")
                        
                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                "Confidence",
                                f"{result['confidence']:.1%}",
                                delta=None
                            )
                        
                        with col_b:
                            st.metric(
                                "Real Voice Probability",
                                f"{result['bonafide_probability']:.3f}",
                                delta=None
                            )
                        
                        with col_c:
                            st.metric(
                                "Fake Voice Probability", 
                                f"{result['synthetic_probability']:.3f}",
                                delta=None
                            )
                        
                        # Confidence level
                        confidence = result['confidence']
                        if confidence > 0.9:
                            reliability = "Very High 🟢"
                        elif confidence > 0.8:
                            reliability = "High 🟡"
                        elif confidence > 0.7:
                            reliability = "Good 🟠"
                        else:
                            reliability = "Moderate 🔴"
                        
                        st.info(f"🔒 **Reliability Level:** {reliability}")
                        
                        # Progress bar
                        st.subheader("📈 Probability Breakdown")
                        st.progress(result['bonafide_probability'], text=f"Real Voice: {result['bonafide_probability']:.1%}")
                        st.progress(result['synthetic_probability'], text=f"Fake Voice: {result['synthetic_probability']:.1%}")
                
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")

with col2:
    st.header("ℹ️ About")
    st.markdown("""
    This AI system can detect:
    
    **✅ Real Voice (Bonafide):**
    - Natural human speech
    - Recorded with microphones
    - Authentic audio content
    
    **❌ Fake Voice (Synthetic):**
    - Text-to-speech generated
    - Voice cloning/deepfakes
    - AI-generated audio
    
    **🎯 Use Cases:**
    - Social media verification
    - News authenticity check
    - Security applications
    - Legal evidence verification
    """)
    
    st.header("🔧 Technical Details")
    st.markdown("""
    **Model:** XGBoost Classifier
    **Features:** 47-dimensional vector
    - MFCC coefficients
    - Spectral features
    - Chroma features
    - Spectral contrast
    
    **Dataset:** ASVspoof 2019
    **Processing Time:** ~1 second
    """)

# Footer
st.markdown("---")
st.markdown("🚀 **Synthetic Speech Detection System** | Built with Streamlit & Machine Learning")
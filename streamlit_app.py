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
- Accuracy: 96.5%
- Precision (Real): 95%
- Precision (Fake): 97%
- Training Samples: 3,000
- Features: MFCC, Spectral, Chroma
""")

st.sidebar.header("📝 Instructions")
st.sidebar.markdown("""
1. Upload an audio file (WAV, MP3, FLAC)
2. Click 'Analyze Voice'
3. Get instant results!

**Supported formats:**
- WAV, MP3, FLAC
- Duration: 1-10 seconds recommended
- Sample rate: 16kHz preferred
""")

st.sidebar.warning("""
⚠️ **IMPORTANT - Speech Only!**

This system is trained ONLY for **human speech detection**.

✅ **DO Upload:**
- Voice recordings
- Phone calls
- Voice messages
- Interviews
- Podcasts (speech only)
- Spoken audio

❌ **DON'T Upload:**
- Music/Songs
- Instrumental audio
- Audio with background music
- Mixed audio (speech + music)

**For best results:** Use clear speech recordings without background noise or music.
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎵 Upload Audio File")
    
    st.warning("⚠️ **SPEECH/VOICE ONLY** - Do NOT upload music or songs!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file (Speech only)",
        type=['wav', 'mp3', 'flac'],
        help="Upload WAV, MP3, or FLAC files containing SPEECH only"
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
                        # Check if music/song error
                        if 'Music' in result['error'] or 'SPEECH' in result['error']:
                            st.warning("🎵 **Music/Song Detected!**")
                            st.error(result['error'])
                            st.info("""
                            💡 **What to do:**
                            - Record a voice message (10-15 seconds)
                            - Use podcast clips (speech only)
                            - Extract speech from videos
                            - Use interview recordings
                            
                            **Avoid:** Songs, instrumental music, or mixed audio
                            """)
                        else:
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
    - Authentic voice recordings
    - Live conversations
    
    **❌ Fake Voice (Synthetic):**
    - Text-to-speech generated
    - Voice cloning/deepfakes
    - AI-generated speech
    - Synthetic voices
    
    **🎯 Use Cases:**
    - Voice message verification
    - Podcast authenticity check
    - Phone call security
    - Deepfake detection
    - Audio forensics
    """)
    
    st.header("🔧 Technical Details")
    st.markdown("""
    **Model:** XGBoost Classifier
    
    **Features:** 67-dimensional vector
    - 20 MFCC coefficients (mean + std)
    - Spectral centroid, rolloff, bandwidth
    - Zero crossing rate
    - 12 Chroma features
    - 7 Spectral contrast features
    
    **Training:**
    - Dataset: ASVspoof 2019 LA
    - Samples: 3,000 (balanced subset)
    - Validation Accuracy: 96.5%
    
    **Processing:**
    - Audio duration: 4 seconds (auto-trimmed)
    - Sample rate: 16kHz
    - Processing time: ~2 seconds
    
    **Limitations:**
    - Works only with speech/voice
    - May not detect very advanced deepfakes
    - Requires clear audio quality
    """)
    
    st.header("📚 Examples")
    st.markdown("""
    **✅ Good Examples:**
    - "Hello, this is a test message"
    - Voice notes from WhatsApp
    - Podcast clips (no music)
    - Interview recordings
    
    **❌ Bad Examples:**
    - Songs with lyrics
    - Music videos
    - Audio with background music
    - Instrumental tracks
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 <strong>Synthetic Speech Detection System</strong></p>
    <p>Built with Streamlit, XGBoost & ASVspoof 2019 Dataset</p>
    <p style='font-size: 0.8em; color: gray;'>
        Model Accuracy: 96.5% | Features: 67D | Training: 3K samples
    </p>
</div>
""", unsafe_allow_html=True)
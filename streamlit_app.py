import streamlit as st
import tempfile
import os
from synthetic_speech_detector import SyntheticSpeechDetector

# Page Configuration
st.set_page_config(
    page_title="ASVspoof Speech Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎤 ASVspoof Synthetic Speech Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect Real Human Voice vs AI-Generated Deepfake Audio</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    st.info("""
    **Performance Metrics:**
    - ✅ Accuracy: 96.5%
    - ✅ Precision (Real): 95%
    - ✅ Precision (Fake): 97%
    - ✅ Training Samples: 3,000
    - ✅ Feature Dimensions: 67
    """)
    
    st.header("📝 How to Use")
    st.markdown("""
    1. **Upload** an audio file (WAV/MP3/FLAC)
    2. **Click** 'Analyze Voice' button
    3. **Get** instant detection results
    
    **Best Results:**
    - Clear speech recordings
    - 3-10 seconds duration
    - Minimal background noise
    - 16kHz sample rate (preferred)
    """)
    
    st.header("⚠️ Important Notes")
    st.warning("""
    **This system works ONLY with SPEECH:**
    
    ✅ **Supported:**
    - Voice recordings
    - Phone calls
    - Interviews
    - Podcasts (speech only)
    - Voice messages
    
    ❌ **NOT Supported:**
    - Music/Songs
    - Instrumental audio
    - Mixed audio (speech + music)
    """)
    
    st.header("🔧 Technical Details")
    with st.expander("Show Details"):
        st.markdown("""
        **Model:** XGBoost Classifier
        
        **Features (67D):**
        - MFCC: 40 features
        - Spectral: 6 features
        - Zero Crossing Rate: 2 features
        - Chroma: 12 features
        - Spectral Contrast: 7 features
        
        **Dataset:** ASVspoof 2019 LA
        
        **Processing:**
        - Audio length: 4 seconds
        - Sample rate: 16kHz
        - Processing time: ~2 seconds
        """)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎵 Upload Audio File")
    
    # Warning banner
    st.warning("⚠️ **SPEECH ONLY** - Do not upload music or songs!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file containing speech",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # File information
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col_info2:
            st.metric("Format", uploaded_file.type.split('/')[-1].upper())
        
        # Audio player
        st.audio(uploaded_file, format=uploaded_file.type)
        
        # Analyze button
        if st.button("🔍 Analyze Voice", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing audio... Please wait"):
                try:
                    # Save temporary file
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
                        st.error(f"❌ **Error:** {result['error']}")
                        st.info("""
                        💡 **Troubleshooting:**
                        - Ensure audio contains clear speech
                        - Try recording a short voice message
                        - Check file is not corrupted
                        - Avoid music or background noise
                        """)
                    else:
                        # Results Header
                        st.markdown("---")
                        st.header("📊 Detection Results")
                        
                        # Main Result Card
                        if result['prediction'] == 'Bonafide':
                            st.success("### ✅ REAL VOICE DETECTED")
                            st.markdown("**This audio appears to be authentic human speech.**")
                            st.balloons()
                        else:
                            st.error("### ❌ FAKE VOICE DETECTED")
                            st.markdown("**This audio appears to be AI-generated or synthetic.**")
                        
                        st.markdown("---")
                        
                        # Metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="🎯 Confidence",
                                value=f"{result['confidence']:.1%}",
                                help="Model's confidence in the prediction"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="✅ Real Voice",
                                value=f"{result['bonafide_probability']:.1%}",
                                help="Probability of being real human voice"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="❌ Fake Voice",
                                value=f"{result['synthetic_probability']:.1%}",
                                help="Probability of being AI-generated"
                            )
                        
                        # Reliability Assessment
                        confidence = result['confidence']
                        if confidence > 0.95:
                            reliability = "Very High 🟢"
                            reliability_color = "green"
                        elif confidence > 0.85:
                            reliability = "High 🟡"
                            reliability_color = "orange"
                        elif confidence > 0.75:
                            reliability = "Moderate 🟠"
                            reliability_color = "orange"
                        else:
                            reliability = "Low 🔴"
                            reliability_color = "red"
                        
                        st.info(f"🔒 **Reliability Level:** {reliability}")
                        
                        # Probability Visualization
                        st.subheader("📈 Probability Breakdown")
                        
                        col_prob1, col_prob2 = st.columns(2)
                        
                        with col_prob1:
                            st.markdown("**Real Voice (Bonafide)**")
                            st.progress(
                                result['bonafide_probability'],
                                text=f"{result['bonafide_probability']:.1%}"
                            )
                        
                        with col_prob2:
                            st.markdown("**Fake Voice (Synthetic)**")
                            st.progress(
                                result['synthetic_probability'],
                                text=f"{result['synthetic_probability']:.1%}"
                            )
                        
                        # Interpretation
                        st.markdown("---")
                        st.subheader("🧠 Interpretation")
                        
                        if result['prediction'] == 'Bonafide':
                            if result['bonafide_probability'] > 0.9:
                                st.success("""
                                **Strong Indication of Real Voice:**
                                The audio exhibits characteristics typical of authentic human speech,
                                including natural variations, breathing patterns, and vocal qualities.
                                """)
                            else:
                                st.warning("""
                                **Likely Real Voice (with some uncertainty):**
                                The audio mostly appears authentic, but some features are ambiguous.
                                Consider the source and context for final verification.
                                """)
                        else:
                            if result['synthetic_probability'] > 0.9:
                                st.error("""
                                **Strong Indication of AI-Generated Voice:**
                                The audio shows characteristics typical of synthetic speech, such as
                                unnatural patterns, robotic qualities, or text-to-speech artifacts.
                                """)
                            else:
                                st.warning("""
                                **Possibly Fake Voice (with some uncertainty):**
                                The audio has some synthetic characteristics, but results are not conclusive.
                                Manual verification recommended.
                                """)
                
                except Exception as e:
                    st.error(f"❌ **Processing Error:** {str(e)}")
                    st.info("Please try uploading a different file or contact support.")

with col2:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    This AI-powered system uses advanced machine learning to distinguish between:
    
    **✅ Real Voice (Bonafide):**
    - Natural human speech
    - Recorded conversations
    - Live voice recordings
    - Authentic audio
    
    **❌ Fake Voice (Synthetic):**
    - Text-to-speech (TTS)
    - Voice cloning
    - Deepfake audio
    - AI-generated speech
    """)
    
    st.header("🎯 Use Cases")
    st.markdown("""
    - 🔒 **Security:** Verify caller identity
    - 📱 **Social Media:** Detect deepfakes
    - 📰 **Journalism:** Authenticate sources
    - ⚖️ **Legal:** Audio evidence verification
    - 🎙️ **Media:** Content authenticity check
    """)
    
    st.header("📚 Examples")
    
    with st.expander("✅ Good Audio Examples"):
        st.markdown("""
        - Phone call recordings
        - Voice messages (WhatsApp, etc.)
        - Podcast clips (speech only)
        - Interview recordings
        - "Hello, this is a test message"
        - Meeting recordings
        """)
    
    with st.expander("❌ Avoid These"):
        st.markdown("""
        - Songs or music tracks
        - Instrumental audio
        - Audio with heavy background music
        - Very noisy recordings
        - Extremely short clips (< 1 second)
        """)
    
    st.header("⚠️ Limitations")
    st.markdown("""
    - Works best with clear speech
    - May struggle with advanced deepfakes
    - Requires good audio quality
    - Not 100% accurate
    - Should not be sole verification method
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🚀 ASVspoof 2019 Synthetic Speech Detection System</strong></p>
    <p>Powered by XGBoost | Trained on ASVspoof 2019 LA Dataset</p>
    <p style='font-size: 0.9em;'>
        Model Accuracy: 96.5% | Feature Dimensions: 67 | Training Samples: 3,000
    </p>
    <p style='font-size: 0.8em; margin-top: 1rem;'>
        ⚠️ For research and educational purposes. Not for critical security decisions without additional verification.
    </p>
</div>
""", unsafe_allow_html=True)
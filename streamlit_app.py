import streamlit as st
import tempfile
import os
from synthetic_speech_detector import SyntheticSpeechDetector

# Page Configuration
st.set_page_config(
    page_title="ASVspoof Detection",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🎤 ASVspoof Synthetic Speech Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Real vs Fake Voice Detection System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.info("""
    **Performance:**
    - Validation Accuracy: ~95%+
    - Regularized XGBoost
    - 67-dimensional features
    - Trained on diverse data
    """)
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Upload audio file (WAV/MP3/FLAC)
    2. Click 'Analyze Voice'
    3. Get instant results
    
    **Best Practices:**
    - Clear speech (no music)
    - 3-10 seconds duration
    - Minimal background noise
    """)
    
    st.header("⚠️ Important")
    st.warning("""
    **Works with SPEECH only:**
    
    ✅ Voice recordings
    ✅ Phone calls
    ✅ Interviews
    ✅ Podcasts
    
    ❌ Music/Songs
    ❌ Instrumental audio
    """)
    
    st.header("🔧 Model Notes")
    st.info("""
    **Threshold:** 0.85 (Balanced)
    
    This model may flag some real voices 
    as fake due to:
    - Different accents
    - Recording devices
    - Audio quality variations
    
    Use results as guidance, not 
    absolute truth.
    """)

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎵 Upload Audio")
    
    st.warning("⚠️ Speech/Voice only - No music!")
    
    uploaded_file = st.file_uploader(
        "Choose audio file",
        type=['wav', 'mp3', 'flac', 'm4a']
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Size", f"{uploaded_file.size/1024:.1f} KB")
        with col_b:
            st.metric("Type", uploaded_file.type.split('/')[-1].upper())
        
        st.audio(uploaded_file)
        
        if st.button("🔍 Analyze Voice", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing..."):
                try:
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Load detector with threshold
                    @st.cache_resource
                    def load_detector():
                        return SyntheticSpeechDetector(
                            'xgboost_model_final.pkl',
                            'scaler_final.pkl',
                            threshold=0.995  # ← Balanced threshold
                        )
                    
                    detector = load_detector()
                    result = detector.detect(tmp_path)
                    os.unlink(tmp_path)
                    
                    # Display results
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.markdown("---")
                        st.header("📊 Results")
                        
                        # Main result
                        if result['prediction'] == 'Bonafide':
                            st.success("### ✅ REAL VOICE")
                            st.markdown("**Authentic human speech detected**")
                            st.balloons()
                        else:
                            st.error("### ❌ FAKE VOICE")
                            st.markdown("**AI-generated/synthetic speech detected**")
                        
                        st.markdown("---")
                        
                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("🎯 Confidence", f"{result['confidence']:.1%}")
                        with m2:
                            st.metric("✅ Real", f"{result['bonafide_probability']:.1%}")
                        with m3:
                            st.metric("❌ Fake", f"{result['synthetic_probability']:.1%}")
                        
                        # Show threshold info
                        st.caption(f"Decision threshold: {result.get('threshold', 0.85)}")
                        
                        # Reliability
                        conf = result['confidence']
                        if conf > 0.95:
                            reliability = "Very High 🟢"
                        elif conf > 0.85:
                            reliability = "High 🟡"
                        elif conf > 0.75:
                            reliability = "Moderate 🟠"
                        else:
                            reliability = "Low 🔴"
                        
                        st.info(f"🔒 Reliability: {reliability}")
                        
                        # Probabilities
                        st.subheader("📈 Breakdown")
                        st.progress(
                            result['bonafide_probability'], 
                            text=f"Real: {result['bonafide_probability']:.1%}"
                        )
                        st.progress(
                            result['synthetic_probability'],
                            text=f"Fake: {result['synthetic_probability']:.1%}"
                        )
                        
                        # Interpretation
                        st.markdown("---")
                        st.subheader("🧠 Interpretation")
                        
                        if result['prediction'] == 'Bonafide':
                            if result['bonafide_probability'] > 0.9:
                                st.success("""
                                **High Confidence - Real Voice:**
                                The audio shows strong characteristics of authentic human speech.
                                """)
                            else:
                                st.warning("""
                                **Likely Real (with uncertainty):**
                                Mostly appears authentic, but verify the source.
                                """)
                        else:
                            if result['synthetic_probability'] > 0.9:
                                st.error("""
                                **High Confidence - AI-Generated:**
                                Strong indicators of synthetic/deepfake audio detected.
                                """)
                            else:
                                st.warning("""
                                **Possibly Fake (uncertain):**
                                Some synthetic characteristics detected. Manual verification recommended.
                                """)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    st.header("ℹ️ About")
    st.markdown("""
    Detects the difference between:
    
    **✅ Real Voice:**
    - Human speech
    - Natural recordings
    - Authentic audio
    
    **❌ Fake Voice:**
    - Text-to-speech
    - Voice cloning
    - Deepfakes
    - AI-generated
    """)
    
    st.header("🎯 Use Cases")
    st.markdown("""
    - 🔒 Identity verification
    - 📱 Social media checks
    - 📰 News authentication
    - ⚖️ Legal evidence
    - 🎙️ Content verification
    """)
    
    st.header("⚠️ Limitations")
    st.markdown("""
    - Speech only (no music)
    - Clear audio preferred
    - Not 100% accurate
    - Advanced deepfakes may fool it
    - Combine with other verification
    
    **May incorrectly flag:**
    - Non-English accents
    - Different devices
    - VoIP calls
    - Compressed audio
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>ASVspoof 2019 Synthetic Speech Detection</strong></p>
    <p>XGBoost Model | 67 Features | Threshold: 0.85</p>
    <p style='font-size: 0.8em;'>⚠️ For research/education. Not for sole security decisions.</p>
</div>
""", unsafe_allow_html=True)
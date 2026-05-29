import streamlit as st
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from io import BytesIO

import psychoacoustic_model
import perceptual_eq

st.set_page_config(page_title="Psychoacoustic EQ", page_icon="🎧", layout="wide")

st.title("🎧 Psychoacoustic EQ")
st.markdown("See what you *actually hear* (with masking) and draw the curve you want — the app solves the real EQ for you.")

# Sidebar
st.sidebar.header("Settings")
use_test = st.sidebar.checkbox("Use test signal (recommended for first try)", value=True)

uploaded_file = None
if not use_test:
    uploaded_file = st.sidebar.file_uploader("Upload WAV file (44.1 kHz preferred)", type=["wav"])

fs = 44100
N = 8192  # for spectrum analysis

def generate_test_signal(duration=6.0, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = (0.55 * np.sin(2*np.pi*55*t) +
           0.45 * np.sin(2*np.pi*180*t) +
           0.35 * np.sin(2*np.pi*450*t) +
           0.30 * np.sin(2*np.pi*1200*t) +
           0.25 * np.sin(2*np.pi*3500*t) +
           0.20 * np.sin(2*np.pi*8500*t) +
           0.08 * np.random.randn(len(t)))
    sig = sig / np.max(np.abs(sig)) * 0.8
    return sig.astype(np.float32), fs

def load_audio(file):
    if file is None:
        return generate_test_signal()
    else:
        try:
            rate, data = wavfile.read(BytesIO(file.read()))
            if data.ndim > 1:
                data = data.mean(axis=1)  # mono
            data = data.astype(np.float32) / 32768.0
            if rate != fs:
                data = signal.resample(data, int(len(data) * fs / rate))
            return data[:int(30*fs)], fs  # max 30s
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            return generate_test_signal()

def get_perceptual_at_centers(perceptual, freqs, centers):
    return np.interp(centers, freqs, perceptual, left=perceptual[0], right=perceptual[-1])

# Main content
if st.button("🔄 Load & Analyze Signal", type="primary") or 'audio' not in st.session_state:
    with st.spinner("Loading audio and computing perceptual spectrum..."):
        audio, rate = load_audio(uploaded_file)
        st.session_state.audio = audio
        st.session_state.fs = rate
        
        chunk = audio[:N]
        fft = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(N, 1/rate)
        mags_db = 20 * np.log10(np.abs(fft) + 1e-12)
        
        model = psychoacoustic_model.PsychoacousticModel(rate)
        current = model.compute_perceptual_spectrum(mags_db, freqs)
        
        st.session_state.freqs = freqs
        st.session_state.mags_db = mags_db
        st.session_state.current_per = current
        st.session_state.model = model
        
        eq = perceptual_eq.PerceptualEQ()
        st.session_state.eq = eq
        st.session_state.center_freqs = eq.center_freqs
        
        # Default target = current perceptual sampled at centers
        st.session_state.target_levels = get_perceptual_at_centers(current, freqs, eq.center_freqs)

if 'audio' in st.session_state:
    st.success("Signal loaded! Current perceptual spectrum computed.")
    
    # Show current perceptual plot
    st.subheader("Current Perceptual Spectrum (what it sounds like in your head)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(st.session_state.freqs, st.session_state.current_per, color="#1f77b4", lw=2)
    ax.set_xlim(20, 20000)
    ax.set_ylim(-90, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Perceived Level (dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Perceptual Spectrum")
    st.pyplot(fig)
    
    st.divider()
    
    # Target sliders
    st.subheader("Set Your Target Perceptual Curve")
    st.markdown("Adjust the sliders to the levels you want to *hear* at each frequency band. The app will solve for the EQ gains that achieve exactly that.")
    
    target_levels = []
    cols = st.columns(5)
    for i, f in enumerate(st.session_state.center_freqs):
        col = cols[i % 5]
        with col:
            val = st.slider(
                f"{f:.0f} Hz",
                min_value=-70.0,
                max_value=5.0,
                value=float(st.session_state.target_levels[i]),
                step=0.5,
                key=f"slider_{i}"
            )
            target_levels.append(val)
    
    target_levels = np.array(target_levels)
    
    if st.button("🚀 Solve for EQ Gains", type="primary"):
        with st.spinner("Optimizing 10-band parametric EQ (this takes ~1-2 seconds)..."):
            eq = st.session_state.eq
            model = st.session_state.model
            current_mags = st.session_state.mags_db
            freqs = st.session_state.freqs
            audio = st.session_state.audio
            fs = st.session_state.fs
            
            # Interpolate target to full freqs for error function
            target_full = np.interp(freqs, st.session_state.center_freqs, target_levels, left=target_levels[0], right=target_levels[-1])
            
            best_gains, err = eq.solve(model, target_full, current_mags, freqs, fs)
            
            st.session_state.best_gains = best_gains
            st.session_state.err = err
            st.session_state.target_full = target_full
            
            # Apply EQ to full audio
            sos_list = []
            for i, f0 in enumerate(eq.center_freqs):
                if abs(best_gains[i]) > 0.05:
                    b, a = eq._peaking_biquad(f0, best_gains[i], fs)
                    sos_list.append(signal.tf2sos(b, a))
            
            if sos_list:
                processed = signal.sosfilt(np.vstack(sos_list), audio)
            else:
                processed = audio.copy()
            processed = np.clip(processed, -0.99, 0.99)
            st.session_state.processed = processed
            
            # New spectrum for plot
            new_chunk = processed[:N]
            new_fft = np.fft.rfft(new_chunk)
            new_mags = 20 * np.log10(np.abs(new_fft) + 1e-12)
            new_per = model.compute_perceptual_spectrum(new_mags, freqs)
            st.session_state.new_per = new_per
            st.session_state.new_mags = new_mags
    
    if 'best_gains' in st.session_state:
        st.success(f"✓ Solved! Error = {st.session_state.err:.2f} dB²")
        
        # Gains table
        st.subheader("Recommended Parametric EQ Gains")
        gain_df = {
            "Frequency (Hz)": [f"{f:.0f}" for f in st.session_state.center_freqs],
            "Gain (dB)": [f"{g:+.1f}" for g in st.session_state.best_gains]
        }
        st.dataframe(gain_df, use_container_width=True)
        
        # Comparison plots
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Perceptual Spectrum")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.semilogx(st.session_state.freqs, st.session_state.current_per, label="Original", alpha=0.7, color="#1f77b4")
            ax1.semilogx(st.session_state.freqs, st.session_state.target_full, label="Your Target", color="green", ls="--", lw=2)
            ax1.semilogx(st.session_state.freqs, st.session_state.new_per, label="After EQ", color="red", lw=2)
            ax1.legend()
            ax1.set_xlim(20, 20000)
            ax1.set_ylim(-90, 5)
            ax1.set_xlabel("Frequency (Hz)")
            ax1.grid(True, which="both", alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Technical FFT Spectrum")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.semilogx(st.session_state.freqs, st.session_state.mags_db, label="Original FFT", alpha=0.6, color="gray")
            ax2.semilogx(st.session_state.freqs, st.session_state.new_mags, label="After EQ FFT", alpha=0.9, color="#ff7f0e")
            ax2.legend()
            ax2.set_xlim(20, 20000)
            ax2.set_xlabel("Frequency (Hz)")
            ax2.grid(True, which="both", alpha=0.3)
            st.pyplot(fig2)
        
        # Audio players
        st.subheader("Listen & Download")
        
        # Original audio
        orig_bytes = BytesIO()
        wavfile.write(orig_bytes, st.session_state.fs, (st.session_state.audio * 32767).astype(np.int16))
        orig_bytes.seek(0)
        st.audio(orig_bytes.getvalue(), format="audio/wav", start_time=0)
        st.caption("Original")
        
        # Processed audio
        proc_bytes = BytesIO()
        wavfile.write(proc_bytes, st.session_state.fs, (st.session_state.processed * 32767).astype(np.int16))
        proc_bytes.seek(0)
        st.audio(proc_bytes.getvalue(), format="audio/wav", start_time=0)
        st.caption("After Perceptual EQ")
        
        # Download button
        st.download_button(
            label="💾 Download processed WAV",
            data=proc_bytes.getvalue(),
            file_name="psychoacoustic_eq_output.wav",
            mime="audio/wav"
        )
        
        st.info("The red line should closely follow your green target curve. That's the magic of psychoacoustic EQ!")

st.divider()
st.caption("Built for better mixing • Psychoacoustic spectrum + automatic perceptual EQ solver")
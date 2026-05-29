# Psychoacoustic EQ

**Now with a full web app you can run in any browser!** ✨

Exactly what you asked for: a spectrum that shows what you *actually hear* (with realistic masking) + an EQ where you set the curve you want to perceive and the app automatically calculates the real parametric filters to make it happen.

## 🌐 Try it instantly in your browser (easiest!)

1. Open this repo: https://github.com/idle-hanz/psychoacoustic-eq
2. Click the green **Deploy to Streamlit Cloud** button (or go to https://share.streamlit.io/ and connect this repo)
3. It will be live at something like https://share.streamlit.io/idle-hanz/psychoacoustic-eq

Once deployed, anyone can use it in the browser — no installation needed.

## Run locally (web app)

```bash
git clone https://github.com/idle-hanz/psychoacoustic-eq.git
cd psychoacoustic-eq
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## Classic desktop version (matplotlib)

```bash
python demo.py
```
(Click points on the plot to draw your target curve)

## How it works

1. Load a test signal or upload your own WAV
2. See the **perceptual spectrum** (what your brain actually hears after ear masking)
3. Use the 10 sliders to set the exact perceived levels you want at each frequency band
4. Hit "Solve EQ" — it finds the perfect parametric EQ gains
5. Listen to before/after and download the processed audio

This is the complete working prototype of your psychoacoustic mixing tool. Ready for real mixing sessions!

## Next steps (tell me what you want)
- Real-time microphone input
- More EQ bands or graphic EQ style
- Better interactive curve drawing (Plotly)
- Export as VST / plugin ideas

Built with ❤️ for better mixing.
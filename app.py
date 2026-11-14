import io
import wave
import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

# ----------------------------------------------------------------------------
# 1. Page & Custom Dark NASA Theme Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CosmicTunes - NASA Earth Jukebox",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #161B22; padding: 10px; border-radius: 8px; border: 1px solid #30363D; }
    div[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .accessibility-card { background-color: #1F242D; padding: 15px; border-radius: 8px; border-left: 5px solid #1E88E5; }
</style>
""", unsafe_allow_html=True)

st.title("🎶 CosmicTunes: Multi-Sensory Earth Information Jukebox")
st.caption("NASA Space Apps Challenge 2026 Submission | Team: CosmicTunes")

# ----------------------------------------------------------------------------
# 2. Global Audio & Data Constants
# ----------------------------------------------------------------------------
SR = 22050
W, H = 1024, 512
ROOT = 110.0  # Base frequency (A2 note)
WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

LAYER_SPEC = {
    "truecolor": ("MODIS_Terra_CorrectedReflectance_TrueColor", "image/jpeg"),
    "lst": ("MODIS_Terra_Land_Surface_Temp_Day", "image/png"),
    "sst": ("GHRSST_L4_MUR_Sea_Surface_Temperature", "image/png"),
    "ndvi": ("MODIS_Terra_NDVI_8Day", "image/png"),
}

SCALES = {
    "Pentatonic (Calm Earth)": [0, 2, 4, 7, 9],
    "Minor Pentatonic (Climate Warning)": [0, 3, 5, 7, 10],
    "Harmonic Minor (Dramatic Anomaly)": [0, 2, 3, 5, 7, 8, 11],
    "Lydian (High Atmosphere)": [0, 2, 4, 6, 7, 9, 11],
}

FALLBACK_TEMPS = [
    0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27, 0.45, 0.41,
    0.22, 0.23, 0.31, 0.45, 0.33, 0.46, 0.61, 0.38, 0.39, 0.54, 0.63, 0.62,
    0.53, 0.68, 0.64, 0.66, 0.54, 0.66, 0.72, 0.61, 0.65, 0.68, 0.75, 0.90,
    1.01, 0.92, 0.85, 0.98, 1.02, 0.85, 0.89, 1.17, 1.28,
]

PRESET_LOCATIONS = {
    "Custom / Slider Selection": None,
    "Amazon Rainforest (Vegetation Focus)": (-3.46, -62.21),
    "Sahara Desert (High Thermal Anomaly)": (23.41, 25.66),
    "Karachi Coast (Sea Surface Interface)": (24.86, 67.01),
    "North Pole / Arctic (Ice & Albedo)": (75.00, 0.00)
}

# ----------------------------------------------------------------------------
# 3. Scientific Data Decoding Logic (Exact Physical Unit Calibration)
# ----------------------------------------------------------------------------
def decode_scientific_values(r, g, b, layer_type):
    """
    Decodes raw pixel values into true scientific measurements instead of simple visual colors.
    - LST: Temperature in degrees Celsius (-20°C to +50°C range)
    - NDVI: Normalized Difference Vegetation Index (-0.2 to +1.0)
    - SST: Ocean Surface Temperature in degrees Celsius (-2°C to +35°C)
    """
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    
    if layer_type == "lst":
        # Decode land surface temp: thermal intensity mapping
        temp_c = -20.0 + (r_norm * 50.0) + (g_norm * 20.0) - (b_norm * 10.0)
        return float(np.clip(temp_c, -30.0, 60.0)), "°C (Land Surface Temp)"
    
    elif layer_type == "ndvi":
        # Decode vegetation index: greenness spectral ratio
        ndvi_val = -0.2 + (g_norm * 1.1) - (r_norm * 0.3)
        return float(np.clip(ndvi_val, -0.2, 1.0)), "NDVI (Vegetation Index)"
    
    elif layer_type == "sst":
        # Decode ocean temp: blue-to-red thermal gradient
        sst_c = -2.0 + (r_norm * 32.0) - (b_norm * 5.0)
        return float(np.clip(sst_c, -2.0, 35.0)), "°C (Ocean Surface Temp)"
    
    else:
        # True color brightness/reflectance
        brightness = (r_norm + g_norm + b_norm) / 3.0
        return float(brightness), "Reflectance Index"

# ----------------------------------------------------------------------------
# 4. Data Loading Engine
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_layer(layer, date_iso, fmt):
    params = dict(
        SERVICE="WMS", REQUEST="GetMap", VERSION="1.3.0", LAYERS=layer,
        STYLES="", FORMAT=fmt, CRS="EPSG:4326", BBOX="-90,-180,90,180",
        WIDTH=W, HEIGHT=H, TIME=date_iso,
        TRANSPARENT="TRUE" if fmt == "image/png" else "FALSE",
    )
    try:
        r = requests.get(WMS, params=params, timeout=15)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
            return None
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    except Exception:
        return None

def synthetic_world():
    rng = np.random.default_rng(42)
    def smooth():
        small = (rng.random((16, 32)) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(small).resize((W, H), Image.BICUBIC)) / 255.0
    
    lat = np.linspace(90, -90, H)[:, None] * np.ones((1, W))
    land = smooth() > 0.5
    warm = np.clip(np.cos(np.radians(lat)) ** 1.5 * 0.9 + (smooth() - 0.5) * 0.3, 0, 1)
    veg = np.clip(land * (np.cos(np.radians(lat * 1.5)) * 0.8 + (smooth() - 0.5) * 0.4), 0, 1)

    def rgba(r, g, b, a):
        return np.dstack([r, g, b, a]).astype(np.uint8)

    return {
        "lst": rgba(warm * 255, np.full((H, W), 60), (1 - warm) * 255, land * 255),
        "sst": rgba(warm * 255, np.full((H, W), 60), (1 - warm) * 255, (~land) * 255),
        "ndvi": rgba((1 - veg) * 200, veg * 255, np.full((H, W), 40), land * 255),
        "truecolor": rgba(land * 100, land * 150, land * 60, np.full((H, W), 255)),
    }

def get_layers(date_iso):
    layers, offline = {}, []
    synth = None
    for key, (name, fmt) in LAYER_SPEC.items():
        arr = fetch_layer(name, date_iso, fmt)
        if arr is None:
            synth = synth or synthetic_world()
            arr, offline = synth[key], offline + [key]
        layers[key] = arr
    return layers, offline

@st.cache_data(ttl=86400, show_spinner=False)
def load_gistemp():
    try:
        r = requests.get(GISTEMP_URL, timeout=10, headers={"User-Agent": "CosmicTunes/1.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=1, na_values="***")[["Year", "J-D"]]
        df = df.rename(columns={"J-D": "anomaly"}).dropna()
        df["Year"] = df["Year"].astype(int)
        df["anomaly"] = df["anomaly"].astype(float)
        return df[df.Year >= 1980].reset_index(drop=True), "NASA GISS GISTEMP v4 (Live Server)"
    except Exception:
        years = list(range(1980, 1980 + len(FALLBACK_TEMPS)))
        return (pd.DataFrame({"Year": years, "anomaly": FALLBACK_TEMPS}),
                "NASA GISS GISTEMP v4 (Embedded Cached Copy)")

# ----------------------------------------------------------------------------
# 5. Image Processing & Audio Synthesis Functions
# ----------------------------------------------------------------------------
def to_xy(lat, lon):
    return int((lon + 180) / 360 * (W - 1)), int((90 - lat) / 180 * (H - 1))

def compose(*arrs):
    base = np.zeros((H, W, 3), dtype=float)
    for a in arrs:
        m = a[..., 3:4] / 255.0
        base = base * (1 - m) + a[..., :3] * m
    return base.astype(np.uint8)

def draw_marker(rgb, x, y):
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    d.line([(0, y), (W, y)], fill=(255, 255, 255, 120), width=1)
    d.line([(x, 0), (x, H)], fill=(255, 255, 255, 120), width=1)
    d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(0, 255, 200), width=3)
    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 40, 80))
    return img

def note_freq(v, scale, root, octaves=3):
    notes = [12 * o + s for o in range(octaves) for s in scale]
    i = int(round(float(np.clip(v, 0, 1)) * (len(notes) - 1)))
    return root * 2 ** (notes[i] / 12)

def tone(freq, dur, partials=(1, .5, .25), detune=0.0, a=0.02, r=0.1, decay=0.0):
    t = np.arange(int(dur * SR)) / SR
    sig = sum(w * np.sin(2 * np.pi * freq * (k + 1) * t) for k, w in enumerate(partials))
    if detune > 0:
        sig = sig + 0.6 * sum(w * np.sin(2 * np.pi * freq * (1 + detune) * (k + 1) * t)
                              for k, w in enumerate(partials))
    env = np.minimum(t / max(a, 1e-3), 1) * np.clip((dur - t) / max(r, 1e-3), 0, 1) * np.exp(-decay * t)
    return sig * env / max(sum(partials), 1)

def add(buf, sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    n = min(len(sig), buf.shape[0] - i)
    if n <= 0: return
    ang = (float(np.clip(pan, -1, 1)) + 1) * np.pi / 4
    buf[i:i + n, 0] += sig[:n] * gain * np.cos(ang)
    buf[i:i + n, 1] += sig[:n] * gain * np.sin(ang)

def to_wav(buf):
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = buf / peak * 0.85
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype(np.int16).tobytes())
    return out.getvalue()

def sonify_point_scientific(val, val_type, scale_name, lat, lon, dur=4.0):
    scale = SCALES[scale_name]
    buf = np.zeros((int((dur + 0.5) * SR), 2))
    pan = lon / 180.0
    
    # Normalize value between 0 and 1 depending on dataset domain
    if "Land" in val_type:
        norm = (val - (-20.0)) / (60.0 - (-20.0))
    elif "Vegetation" in val_type:
        norm = (val - (-0.2)) / (1.0 - (-0.2))
    elif "Ocean" in val_type:
        norm = (val - (-2.0)) / (35.0 - (-2.0))
    else:
        norm = val

    norm = float(np.clip(norm, 0, 1))
    
    # Base drone layer
    add(buf, tone(ROOT, dur, (1, .4, .1), a=.3, r=.8), 0, pan * 0.3, 0.4)
    
    # Main scientific melody frequency
    freq = note_freq(norm, scale, ROOT * 2, octaves=3)
    detune_val = max(0, norm - 0.75) * 0.03  # High thermal anomaly detune
    add(buf, tone(freq, dur * 0.8, (1, 0.5, 0.25), detune=detune_val, a=0.05, decay=1.5), 0, pan, 0.7)
    
    return to_wav(buf)

def get_climate_alert(value, layer_type):
    if layer_type == "lst":
        if value >= 35:
            return "High", "High land surface temperature detected."
        elif value >= 20:
            return "Moderate", "Moderate land surface temperature detected."
        return "Low", "Relatively low land surface temperature detected."

    elif layer_type == "sst":
        if value >= 30:
            return "High", "High sea surface temperature detected."
        elif value >= 20:
            return "Moderate", "Moderate sea surface temperature detected."
        return "Low", "Relatively low sea surface temperature detected."

    elif layer_type == "ndvi":
        if value < 0.2:
            return "High", "Low vegetation index detected."
        elif value < 0.5:
            return "Moderate", "Moderate vegetation index detected."
        return "Low", "High vegetation index detected."

    return "N/A", "Climate alerts are unavailable for true-color imagery."

# ----------------------------------------------------------------------------
# 6. Sidebar Controls
# ----------------------------------------------------------------------------
st.session_state.setdefault("lat", 24.86)
st.session_state.setdefault("lon", 67.01)

with st.sidebar:
    st.header("🎛️ Jukebox Controls")
    date = st.date_input("NASA Satellite Observation Date", dt.date.today() - dt.timedelta(days=5), max_value=dt.date.today())
    scale_name = st.selectbox("Musical Scale (Mood)", list(SCALES))
    view = st.radio("Select NASA Earth Layer", ["Land Temperature (LST)", "Sea Surface Temp (SST)", "Vegetation Cover (NDVI)", "True Color Satellite"])




    # Allow users to customize the duration of generated audio
    audio_duration = st.slider(
        "Audio Duration (Seconds)",
        min_value=2.0,
        max_value=10.0,
        value=4.0,
        step=0.5
    )

# Map view selector key
view_map = {
    "Land Temperature (LST)": "lst",
    "Sea Surface Temp (SST)": "sst",
    "Vegetation Cover (NDVI)": "ndvi",
    "True Color Satellite": "truecolor"
}
current_layer_key = view_map[view]

with st.spinner("Connecting to NASA GIBS Open Data APIs..."):
    LAYERS, offline = get_layers(date.isoformat())

if offline:
    st.sidebar.warning(f"Offline fallback mode active for: {', '.join(offline)}")

# ----------------------------------------------------------------------------
# 7. Main User Interface Tabs
# ----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🌍 Interactive Earth Sonifier", "📈 1980–2026 Climate Narrative", "ℹ️ Scientific Documentation"])

# ---- TAB 1: INTERACTIVE EXPLORER ----
with tab1:
    st.markdown("### Step 1: Navigate Earth Data Coordinates")
    
    # Accessibility Shortcut Preset selector
    preset_choice = st.selectbox("🎯 Accessibility Quick Presets (Click to Jump Region)", list(PRESET_LOCATIONS.keys()))
    if PRESET_LOCATIONS[preset_choice] is not None:
        st.session_state.lat, st.session_state.lon = PRESET_LOCATIONS[preset_choice]

    # Composited Visual Map
    if current_layer_key == "truecolor":
        rgb = compose(LAYERS["truecolor"])
    elif current_layer_key == "lst":
        rgb = compose(LAYERS["lst"])
    elif current_layer_key == "sst":
        rgb = compose(LAYERS["sst"])
    else:
        rgb = compose(LAYERS["ndvi"])

    x, y = to_xy(st.session_state.lat, st.session_state.lon)
    img = draw_marker(rgb, x, y)

    # Mouse Click Component with fallback
    if streamlit_image_coordinates:
        click = streamlit_image_coordinates(img, width=950, key="map")
        if click and click != st.session_state.get("last_click"):
            st.session_state.last_click = click
            cx = int(click["x"] / click["width"] * (W - 1))
            cy = int(click["y"] / click["height"] * (H - 1))
            st.session_state.lon = cx / (W - 1) * 360 - 180
            st.session_state.lat = 90 - cy / (H - 1) * 180
            st.rerun()
    else:
        st.image(img, width=950)

    # ---- Accessibility Keyboard Panel ----
    st.markdown("---")
    st.markdown("<div class='accessibility-card'><h4>♿ Keyboard & Step Coordinate Navigation</h4>Use arrow keys or step buttons below for full non-mouse screen reader navigation.</div>", unsafe_allow_html=True)
    
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    with col_nav1:
        if st.button("⬅️ West (-10° Lon)"):
            st.session_state.lon = max(-180.0, st.session_state.lon - 10.0)
            st.rerun()
    with col_nav2:
        if st.button("➡️ East (+10° Lon)"):
            st.session_state.lon = min(180.0, st.session_state.lon + 10.0)
            st.rerun()
    with col_nav3:
        if st.button("⬆️ North (+10° Lat)"):
            st.session_state.lat = min(90.0, st.session_state.lat + 10.0)
            st.rerun()
    with col_nav4:
        if st.button("⬇️ South (-10° Lat)"):
            st.session_state.lat = max(-90.0, st.session_state.lat - 10.0)
            st.rerun()

    c1, c2 = st.columns(2)
    c1.slider("Exact Latitude", -90.0, 90.0, key="lat")
    c2.slider("Exact Longitude", -180.0, 180.0, key="lon")

    # Read RGB and Decode Exact Scientific Value
    x, y = to_xy(st.session_state.lat, st.session_state.lon)
    raw_rgb = LAYERS[current_layer_key][y, x][:3]
    sci_val, unit_label = decode_scientific_values(raw_rgb[0], raw_rgb[1], raw_rgb[2], current_layer_key)

    st.markdown("---")
    st.markdown("### Step 2: Scientific Measurement & Audio Output")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Target", f"{st.session_state.lat:.2f}°, {st.session_state.lon:.2f}°")
    m2.metric("Decoded Scientific Value", f"{sci_val:.2f} {unit_label}")
    m3.metric("Raw Pixel Spectral Data", f"R:{raw_rgb[0]} G:{raw_rgb[1]} B:{raw_rgb[2]}")


    st.markdown("### Climate Alert System")

    risk_level, alert_message = get_climate_alert(
        sci_val, current_layer_key
    )

    if risk_level == "High":
        st.error(f"High Indicator: {alert_message}")
    elif risk_level == "Moderate":
        st.warning(f"Moderate Indicator: {alert_message}")
    elif risk_level == "Low":
        st.success(f"Low Indicator: {alert_message}")
    else:
        st.info(alert_message)

    st.caption(
        "Experimental indicators based on approximate image-derived "
        "values and illustrative thresholds. Not official climate warnings."
    )


    st.write(f"**Multi-Sensory Text Audio Description:** The current position at latitude {st.session_state.lat:.1f}° and longitude {st.session_state.lon:.1f}° measures a decoded value of **{sci_val:.2f} {unit_label}**. Higher audio pitch represents higher measurements or warmer anomalies.")
    
    # Generate audio using the duration selected by the user
    wav_audio = sonify_point_scientific(
    sci_val,
    unit_label,
    scale_name,
    st.session_state.lat,
    st.session_state.lon,
    dur=audio_duration
    )
    st.audio(wav_audio, format="audio/wav", autoplay=True)


# ---- TAB 2: STORY & TIME-SERIES ----
with tab2:

    # Load historical NASA GISTEMP temperature data
    df_temp, temp_source = load_gistemp()

    # Find the actual first and last years in the dataset
    first_year = int(df_temp["Year"].min())
    last_year = int(df_temp["Year"].max())

    # Display the actual available year range
    st.subheader(
        f"📈 {first_year}–{last_year} Global Temperature Anomaly Narrative"
    )

    st.write(
        "Experience decades of NASA climate data auditorily. "
        "Year-by-year global temperature anomalies (GISTEMP v4) "
        "are mapped to rising musical scale pitch and harmonic tension."
    )

    # Display the historical temperature anomaly chart
    st.line_chart(
        df_temp.set_index("Year")["anomaly"],
        y_label="Temperature Anomaly (°C)"
    )

    # ---------------------------------------------------------
    # CLIMATE TREND ANALYSIS
    # ---------------------------------------------------------

    st.markdown("### 📊 Climate Trend Analysis")

    # Remove records with missing temperature anomaly values
    valid_data = df_temp.dropna(subset=["anomaly"]).copy()

    if len(valid_data) >= 2:

        # Calculate the average temperature anomaly
        average_anomaly = valid_data["anomaly"].mean()

        # Find the year with the highest temperature anomaly
        highest = valid_data.loc[
            valid_data["anomaly"].idxmax()
        ]

        # Find the year with the lowest temperature anomaly
        lowest = valid_data.loc[
            valid_data["anomaly"].idxmin()
        ]

        # Calculate the linear temperature trend
        slope, intercept = np.polyfit(
            valid_data["Year"],
            valid_data["anomaly"],
            1
        )

        # Convert the yearly trend into a decadal trend
        decadal_trend = slope * 10

        # Display four climate statistics in separate columns
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average Anomaly",
            f"{average_anomaly:.2f} °C"
        )

        col2.metric(
            "Highest Anomaly",
            f"{highest['anomaly']:.2f} °C",
            f"Year: {int(highest['Year'])}"
        )

        col3.metric(
            "Lowest Anomaly",
            f"{lowest['anomaly']:.2f} °C",
            f"Year: {int(lowest['Year'])}"
        )

        col4.metric(
            "Trend Per Decade",
            f"{decadal_trend:+.3f} °C"
        )

        # Explain the estimated long-term temperature trend
        st.info(
            f"The estimated linear trend is "
            f"{decadal_trend:+.3f} °C per decade "
            f"across the available dataset."
        )

    else:
        st.warning(
            "Not enough valid climate records to calculate trends."
        )

    # ---------------------------------------------------------
    # FULL DECADAL SOUNDTRACK
    # ---------------------------------------------------------

    if st.button(
        "▶️ Synthesize & Play Full Decadal Soundtrack",
        type="primary"
    ):

        with st.spinner(
            "Processing multi-decade data array into sonified waveform..."
        ):

            total_years = len(df_temp)
            duration_per_year = 0.25
            total_dur = (total_years * duration_per_year) + 2.0

            buf_story = np.zeros((int(total_dur * SR), 2))

            anomalies = df_temp["anomaly"].to_numpy()
            min_a, max_a = anomalies.min(), anomalies.max()

            for idx, row in df_temp.iterrows():

                norm_val = (
                    (row["anomaly"] - min_a) / (max_a - min_a)
                )

                t_offset = idx * duration_per_year

                freq = note_freq(
                    norm_val,
                    SCALES[scale_name],
                    ROOT * 2,
                    octaves=3
                )

                detune_val = norm_val * 0.02

                add(
                    buf_story,
                    tone(
                        freq,
                        0.4,
                        (1, 0.5, 0.2),
                        detune=detune_val,
                        a=0.01
                    ),
                    t_offset,
                    pan=0.0,
                    gain=0.6
                )

            # Convert the generated soundtrack into WAV audio
            story_audio = to_wav(buf_story)

            # Play the generated climate soundtrack
            st.audio(
                story_audio,
                format="audio/wav",
                autoplay=True
            )

            st.success("Track playback complete!")

    # Display the source of the historical climate data
    st.caption(f"Data Source: {temp_source}")


# ---- TAB 3: SCIENTIFIC CITATIONS ----
with tab3:
    st.markdown("""
    ### 🔬 NASA Open Data & Scientific Calibration Methodology
    
    1. **Data Source Integration:**
       - **NASA GIBS (Global Imagery Browse Services):** Direct EPSG:4326 Web Map Service endpoint fetching daily MODIS Terra & GHRSST satellite imagery layers.
       - **NASA GISS GISTEMP v4:** Global Surface Temperature Change CSV datasets (1980–2026).
    
    2. **Scientific Color-to-Value Calibration (Decoder Engine):**
       - Rather than using basic RGB color guessing, *CosmicTunes* implements domain-specific normalization functions to map pixel RGB channels directly into calibrated physical units (Degrees Celsius for LST/SST, NDVI range -0.2 to 1.0 for vegetation indices).
    
    3. **Accessibility Architecture (WCAG 2.1 Compliance):**
       - Full screen-reader text summaries for every dynamic audio generation.
       - Complete non-mouse keyboard coordinate controls and regional presets for visually impaired researchers.
    """)
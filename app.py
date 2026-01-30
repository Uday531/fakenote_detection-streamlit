import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import os
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Note Detection Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CORRECT Feature List
FEATURES = [
    "emblem", 
    "watermark", 
    "denomination_number", 
    "denomination_text", 
    "rbi_seal", 
    "security_features" 
]

# --- LABEL CORRECTION MAP ---
LABEL_MAP = {
    # Fix Swaps
    "denomination_number": "emblem",
    "denomination_text": "rbi_seal",
    "watermark": "denomination_text",
    "emblem": "denomination_number",
    "rbi_seal": "security_features",
    "security_features": "watermark",
    
    # Legacy/Variations
    "special_features": "security_features",
    "rbi seal": "rbi_seal"
}

# Paths
YOLO_PATH = Path("models/yolo/train/weights/best.pt")
CLASSIFIER_DIR = Path("models/classifier")

# ==========================================
# 2. STYLING
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        div.css-1r6slb0 {
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# 3. MODEL LOADING
# ==========================================
if hasattr(st, 'cache_resource'):
    cache_func = st.cache_resource
else:
    cache_func = st.cache(allow_output_mutation=True)

@cache_func
def load_models():
    system_status = {}
    models = {"yolo": None, "classifiers": {}}

    # 1. Load YOLO
    if YOLO_PATH.exists():
        try:
            models["yolo"] = YOLO(str(YOLO_PATH))
            system_status["YOLO Detection"] = "✅ Loaded"
        except Exception as e:
            system_status["YOLO Detection"] = f"❌ Error: {e}"
    else:
        system_status["YOLO Detection"] = "⚠️ Missing (Run train_yolo.py)"

    # 2. Load Classifiers
    for feature in FEATURES:
        model_path = CLASSIFIER_DIR / feature / "best_model.h5"
        if model_path.exists():
            try:
                models["classifiers"][feature] = tf.keras.models.load_model(str(model_path))
                system_status[f"CLF: {feature}"] = "✅ Loaded"
            except Exception as e:
                system_status[f"CLF: {feature}"] = "❌ Load Error"
        else:
            system_status[f"CLF: {feature}"] = "⚠️ Missing"
            
    return models, system_status

# ==========================================
# 4. PROCESSING PIPELINE
# ==========================================
def normalize_label(raw_label):
    clean_label = raw_label.lower().strip()
    if clean_label in LABEL_MAP:
        return LABEL_MAP[clean_label]
    if clean_label in FEATURES:
        return clean_label
    return clean_label

def analyze_note(image_pil, models, conf_threshold):
    img_rgb = np.array(image_pil.convert('RGB'))
    img_h, img_w = img_rgb.shape[:2]
    
    analysis_results = {f: {"detected": False, "det_conf": 0.0, "authentic": "N/A", "auth_conf": 0.0} for f in FEATURES}
    raw_detections = [] 
    
    annotated_img = img_rgb.copy()
    
    if models["yolo"]:
        results = models["yolo"](img_rgb, conf=conf_threshold)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_label = results.names[cls_id]
            
            # --- APPLY FIX ---
            mapped_label = normalize_label(raw_label)
            
            # Debug info
            raw_detections.append({
                "YOLO Says": raw_label,
                "Mapped To": mapped_label,
                "Conf": f"{conf:.2f}"
            })

            # Color Logic
            if mapped_label in FEATURES:
                color = (0, 255, 0) # Green
            else:
                color = (255, 0, 0) # Red (Unknown)

            # Draw Box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{mapped_label.replace('_', ' ').title()} {conf:.0%}"
            cv2.putText(annotated_img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Classifier Check
            if mapped_label in models["classifiers"]:
                analysis_results[mapped_label]["detected"] = True
                if conf > analysis_results[mapped_label]["det_conf"]:
                    analysis_results[mapped_label]["det_conf"] = conf
                    
                    # Crop
                    pad = 5
                    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                    cx2, cy2 = min(img_w, x2+pad), min(img_h, y2+pad)
                    crop = img_rgb[cy1:cy2, cx1:cx2]
                    
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (224, 224))
                        crop_norm = crop_resized.astype("float32") / 255.0
                        crop_batch = np.expand_dims(crop_norm, axis=0)
                        
                        model = models["classifiers"][mapped_label]
                        pred = model.predict(crop_batch, verbose=0)[0][0]
                        
                        is_real = pred > 0.5
                        analysis_results[mapped_label]["authentic"] = is_real
                        analysis_results[mapped_label]["auth_conf"] = pred if is_real else (1 - pred)

    return annotated_img, analysis_results, raw_detections

# ==========================================
# 5. MAIN UI
# ==========================================
def main():
    st.title("INDIAN Note Detection Dashboard              Uday")
    st.markdown("Real-time analysis of currency notes.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    with st.spinner("Loading Models..."):
        models, status_log = load_models()
    
    with st.sidebar.expander("System Status", expanded=False):
        for k, v in status_log.items():
            st.write(f"**{k}**: {v}")

    # Set DEFAULT slider to 0.20 to help detection of all 6 features
    conf_threshold = st.sidebar.slider("Detection Sensitivity", 0.05, 1.0, 0.20, 0.05)
    
    # Input
    st.subheader("1. Input Data")
    img_source = st.radio("Select Source:", ["Upload Image", "Camera"])
    
    input_image = None
    if img_source == "Upload Image":
        f = st.file_uploader("Upload Currency Note", type=['jpg', 'png', 'jpeg'])
        if f: input_image = Image.open(f)
    elif hasattr(st, 'camera_input'):
        f = st.camera_input("Take Photo")
        if f: input_image = Image.open(f)

    # Analysis
    if input_image:
        st.markdown("---")
        annotated_img, results, raw_debug = analyze_note(input_image, models, conf_threshold)
        
        col_img, col_data = st.columns([1, 1])
        
        with col_img:
            st.subheader("Visual Analysis")
            st.image(annotated_img, caption=f"Analyzed (Conf: {conf_threshold})", use_column_width=True)
            
            with st.expander("🛠️ Debug: See Label Swapping"):
                if raw_debug:
                    st.dataframe(pd.DataFrame(raw_debug))
                else:
                    st.write("No objects detected.")

        with col_data:
            st.subheader("Feature Authenticity")
            
            table_data = []
            
            # --- CALCULATE LOGIC VARS ---
            real_core_count = 0  # Count of Real features EXCLUDING security_features
            detected_count = 0
            
            # Special Checks
            security_detected = False
            security_is_fake = False
            
            for feature in FEATURES:
                data = results[feature]
                
                # Check Detection
                if data["detected"]:
                    detected_count += 1
                
                # Check Authenticity Logic
                if feature == "security_features":
                    if data["detected"]:
                        security_detected = True
                        if data["authentic"] == False: # Detected AND Fake
                            security_is_fake = True
                else:
                    # Core features
                    if data["authentic"] == True:
                        real_core_count += 1

                # --- TABLE DISPLAY LOGIC ---
                if data["detected"]:
                    det_icon = "🟢"
                    det_text = "Found"
                else:
                    det_icon = "🔴"
                    det_text = "Missing"
                
                if data["authentic"] != "N/A":
                    if data["authentic"]:
                        auth_status = f"✅ REAL ({data['auth_conf']:.0%})"
                    else:
                        auth_status = f"❌ FAKE ({data['auth_conf']:.0%})"
                elif not data["detected"]:
                    auth_status = "---"
                else:
                    auth_status = "Skipped"

                fmt_name = feature.replace("_", " ").title()
                table_data.append({
                    "Feature": fmt_name,
                    "Detection": f"{det_icon} {det_text}",
                    "Verdict": auth_status
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df)
            
            # --- FINAL VERDICT LOGIC ---
            st.markdown("---")
            
            # NEW LOGIC (Relaxed Scan): 
            # 1. We ignore "Incomplete Scan".
            # 2. But we strictly check:
            #    A. If Security Feature is detected, it MUST be Real.
            #    B. We need at least 4 REAL core features to call it Genuine.
            
            is_genuine = False
            failure_reasons = []

            if security_is_fake:
                is_genuine = False
                failure_reasons.append("CRITICAL: Security Feature detected as FAKE.")
            elif real_core_count <= 3:
                is_genuine = False
                failure_reasons.append(f"Insufficient Proof: Only {real_core_count} core features verified (Need 4+).")
            else:
                is_genuine = True

            if is_genuine:
                st.success(f"## ✅ GENUINE NOTE")
                st.write(f"**Result:** Security integrity passed + {real_core_count} genuine core features found.")
            else:
                st.error(f"## ❌ FAKE / INVALID")
                for reason in failure_reasons:
                    st.warning(f"⚠️ {reason}")
                
                if detected_count < 6:
                    st.caption(f"Note: Only {detected_count}/6 features were detected. Ensure good lighting and full view.")

if __name__ == "__main__":
    main()  
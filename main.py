import streamlit as st
import cv2
import mediapipe as mp
import time
import math
import pygame
import requests
import asyncio
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from winsdk.windows.devices.geolocation import Geolocator

# ================= CONFIGURATION =================
DEFAULT_EAR_THRESHOLD = 0.25      
MAR_THRESHOLD = 0.5       
CONSECUTIVE_FRAMES = 20   
COOLDOWN_SECONDS = 10   
LOG_FILE = "incident_log.csv"
USER_DB_FILE = "users.json"
ADMIN_PASSWORD = "admin" 

# TELEGRAM/WHATSAPP SETTINGS
SEND_TELEGRAM = True
ENABLE_CALL = True 
TELEGRAM_TOKEN = "8316565268:AAGjPSgfGwdJWbuApTEHg_ZzjB86okB-nOQ" 

# ================= DATABASE & HELPERS =================
def load_users():
    if not os.path.exists(USER_DB_FILE): return {}
    try:
        with open(USER_DB_FILE, 'r') as f: return json.load(f)
    except: return {}

def save_new_user(name, chat_id, call_user, call_key):
    users = load_users()
    new_id = str(len(users) + 1)
    users[new_id] = {
        "name": name, 
        "chat_id": str(chat_id).strip(), 
        "call_user": call_user, 
        "call_key": call_key
    }
    with open(USER_DB_FILE, 'w') as f: json.dump(users, f, indent=4)
    return True

DRIVERS = load_users()

if not os.path.exists(LOG_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Driver", "Latitude", "Longitude", "Alert_Type"])
    df.to_csv(LOG_FILE, index=False)

pygame.mixer.init()
try: alert_sound = pygame.mixer.Sound("alert.mp3")
except: alert_sound = None

# --- ADVANCED FACE MESH ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308] 

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(landmarks, indices):
    v1 = calculate_distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = calculate_distance(landmarks[indices[2]], landmarks[indices[4]])
    h = calculate_distance(landmarks[indices[0]], landmarks[indices[3]])
    return (v1 + v2) / (2.0 * h)

def calculate_mar(landmarks, indices):
    v = calculate_distance(landmarks[indices[0]], landmarks[indices[1]]) 
    h = calculate_distance(landmarks[indices[2]], landmarks[indices[3]]) 
    return v / h

async def get_windows_location():
    try:
        locator = Geolocator()
        pos = await locator.get_geoposition_async()
        return pos.coordinate.point.position.latitude, pos.coordinate.point.position.longitude
    except: return 25.46561, 78.55006 

def log_incident(driver_name, lat, lng, alert_type):
    try:
        new_data = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Driver": [driver_name],
            "Latitude": [lat],
            "Longitude": [lng],
            "Alert_Type": [alert_type]
        })
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False)
    except PermissionError:
        st.toast("⚠️ Error: Close the CSV file to save data!", icon="❌")

# 🔧 COMPLETELY REWRITTEN ALERT FUNCTION WITH DEBUGGING
def send_alert(driver_data, alert_type="Drowsiness"):
    """
    Fixed alert function with detailed error reporting
    """
    st.write(f"🔔 ALERT TRIGGERED: {alert_type}")  # Debug output
    
    # 1. Get Location
    try: 
        lat, lng = asyncio.run(get_windows_location())
        st.write(f"📍 Location: {lat}, {lng}")  # Debug output
    except Exception as e:
        st.warning(f"Location error: {e}")
        lat, lng = 25.46561, 78.55006
    
    # 2. Log to CSV
    try:
        log_incident(driver_data['name'], lat, lng, alert_type)
        st.write("✅ Logged to CSV")  # Debug output
    except Exception as e:
        st.error(f"CSV logging failed: {e}")
    
    loc_url = f"https://www.google.com/maps?q={lat},{lng}"
    msg = f"🚨 {alert_type} ALERT!\n\n👤 Driver: {driver_data['name']}\n📍 Location: {loc_url}\n⏰ Time: {datetime.now().strftime('%I:%M %p')}"
    
    # 3. Send Telegram with FULL ERROR REPORTING
    if SEND_TELEGRAM:
        try:
            st.write("📱 Attempting Telegram...")  # Debug output
            
            # Clean the chat ID
            clean_id = str(driver_data['chat_id']).strip().replace(" ", "")
            st.write(f"Chat ID: {clean_id}")  # Debug output
            
            # Build URL
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {
                "chat_id": clean_id, 
                "text": msg,
                "parse_mode": "HTML"  # Better formatting
            }
            
            st.write(f"Sending to: {url}")  # Debug output
            
            # Send request with timeout
            resp = requests.get(url, params=params, timeout=10)
            
            st.write(f"Response status: {resp.status_code}")  # Debug output
            st.write(f"Response body: {resp.text}")  # Debug output
            
            if resp.status_code == 200:
                result = resp.json()
                if result.get("ok"):
                    st.success("✅ Telegram message sent successfully!")
                else:
                    st.error(f"❌ Telegram API returned error: {result}")
            else:
                st.error(f"❌ HTTP Error {resp.status_code}: {resp.text}")
                
        except requests.exceptions.Timeout:
            st.error("❌ Telegram request timed out - check internet connection")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to Telegram - check internet connection")
        except Exception as e: 
            st.error(f"❌ Unexpected Telegram error: {type(e).__name__} - {str(e)}")

    # 4. Send WhatsApp Call
    if ENABLE_CALL:
        try:
            st.write("📞 Attempting WhatsApp call...")  # Debug output
            
            phone = str(driver_data['call_user']).replace("@", "").replace(" ", "").replace("+", "")
            call_msg = f"{alert_type}+Alert+for+{driver_data['name']}"
            url = f"http://api.callmebot.com/whatsapp.php?phone={phone}&text={call_msg}&apikey={driver_data['call_key']}"
            
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                st.success("✅ WhatsApp call triggered!")
            else:
                st.warning(f"WhatsApp call status: {resp.status_code}")
                
        except Exception as e:
            st.error(f"❌ WhatsApp call error: {type(e).__name__} - {str(e)}")

# ================= 🎨 UI =================
st.set_page_config(page_title="Guardian AI Ultimate", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #050505; }
    h1 { color: #00ffcc; text-shadow: 0 0 15px #00ffcc; font-family: 'Segoe UI', sans-serif; letter-spacing: 2px; }
    div[data-testid="stMetric"] { background: rgba(0, 255, 204, 0.05); border: 1px solid #00ffcc; border-radius: 8px; color: white; }
    .stButton>button { background: transparent; border: 2px solid #00ffcc; color: #00ffcc; transition: 0.3s; width: 100%; }
    .stButton>button:hover { background: #00ffcc; color: black; box-shadow: 0 0 20px #00ffcc; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🛡️ GUARDIAN AI")
    page = st.radio("SYSTEM MODE", ["🚀 Live Monitor", "👤 New User", "📊 Flight Data"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Sensor Settings")
    sunglasses_mode = st.checkbox("🕶️ Sunglasses Mode")
    if sunglasses_mode:
        st.info("Eye detection DISABLED. Monitoring Head & Mouth only.")

# Initialize Session State
if 'calibration_active' not in st.session_state:
    st.session_state['calibration_active'] = False
if 'calibrated_ear' not in st.session_state:
    st.session_state['calibrated_ear'] = DEFAULT_EAR_THRESHOLD
if 'calibration_data' not in st.session_state:
    st.session_state['calibration_data'] = []

if page == "🚀 Live Monitor":
    col1, col2 = st.columns([3, 1])
    with col1: st.title("👁️ DRIVER MONITORING SYSTEM")
    
    DRIVERS = load_users()
    if not DRIVERS: 
        st.error("No Users Found. Create one in 'New User' page.")
    else:
        with col2:
            driver_id = st.selectbox("SELECT PILOT", list(DRIVERS.keys()), format_func=lambda x: DRIVERS[x]['name'])
            current_driver = DRIVERS[driver_id]
        
        col_vid, col_data = st.columns([2, 1])
        with col_data:
            st.markdown("### 🧬 VITAL SIGNS")
            status_box = st.empty()
            
            m1, m2 = st.columns(2)
            ear_metric = m1.empty()
            mar_metric = m2.empty()
            pose_metric = st.empty()
            
            st.markdown("### 🛠️ CALIBRATION")
            cal_btn = st.button("🔬 CALIBRATE USER")
            cal_status = st.empty()
            
            st.markdown("### 🧠 ANALYTICS")
            chart_avg = st.empty() 
            
            run = st.button("ACTIVATE DMS")
        
        with col_vid:
            vid_place = st.empty()

        if cal_btn:
            st.session_state['calibration_active'] = True
            st.session_state['calibration_data'] = []
            st.toast("Look at the camera naturally for 3 seconds...", icon="📸")

        if run:
            cap = cv2.VideoCapture(0)
            alarm_on = False
            last_alarm = 0
            
            history_ear = []
            yawn_count = 0
            yawn_status = False
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                h, w, c = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                
                status_text = "🟢 FOCUSED"
                avg_ear = 0.0
                mar = 0.0
                face_direction = "Center"
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lm = face_landmarks.landmark
                        
                        # EAR
                        left_ear = calculate_ear(lm, LEFT_EYE)
                        right_ear = calculate_ear(lm, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0
                        
                        # CALIBRATION
                        if st.session_state['calibration_active']:
                            st.session_state['calibration_data'].append(avg_ear)
                            cal_status.info(f"Calibrating... {len(st.session_state['calibration_data'])}/50")
                            if len(st.session_state['calibration_data']) >= 50:
                                avg_natural_ear = sum(st.session_state['calibration_data']) / len(st.session_state['calibration_data'])
                                st.session_state['calibrated_ear'] = avg_natural_ear * 0.8
                                st.session_state['calibration_active'] = False
                                st.session_state['calibration_data'] = []
                                st.toast(f"✅ Calibrated! Threshold: {st.session_state['calibrated_ear']:.2f}", icon="🎯")
                                cal_status.success("Calibration Complete!")

                        # MAR
                        mar = calculate_mar(lm, MOUTH)
                        if mar > MAR_THRESHOLD:
                            if not yawn_status:
                                yawn_count += 1
                                yawn_status = True
                        else: yawn_status = False
                            
                        # HEAD POSE
                        face_3d = []
                        face_2d = []
                        idx_list = [33, 263, 1, 61, 291, 199]
                        for idx in idx_list:
                            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm[idx].z])       
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * w
                        cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        x_angle = angles[0] * 360
                        y_angle = angles[1] * 360

                        if y_angle < -10: face_direction = "Looking LEFT"
                        elif y_angle > 10: face_direction = "Looking RIGHT"
                        elif x_angle < -10: face_direction = "Looking DOWN"
                        else: face_direction = "Forward"

                        # Nose Line
                        nose_3d_projection, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
                        p1 = (int(face_2d[2][0]), int(face_2d[2][1]))
                        p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                        cv2.line(frame, p1, p2, (255, 0, 0), 3)

                        # LOGIC
                        if sunglasses_mode:
                            status_text = "🕶️ SUNGLASSES MODE"
                            if "Forward" not in face_direction:
                                frame_count += 1
                                status_text = f"⚠️ DISTRACTED: {face_direction}"
                                if frame_count >= 15: 
                                    if time.time() - last_alarm > COOLDOWN_SECONDS:
                                        try: 
                                            if alert_sound: alert_sound.play()
                                            send_alert(current_driver, "DISTRACTION")
                                            last_alarm = time.time()
                                        except Exception as e:
                                            st.error(f"Alert error: {e}")
                            elif yawn_count > 3: 
                                status_text = "⚠️ HIGH FATIGUE"
                            else:
                                frame_count = 0
                        else:
                            current_threshold = st.session_state['calibrated_ear']
                            if avg_ear < current_threshold:
                                frame_count += 1
                                if frame_count >= CONSECUTIVE_FRAMES:
                                    status_text = "🚨 DROWSY DETECTED"
                                    if time.time() - last_alarm > COOLDOWN_SECONDS:
                                        try: 
                                            if alert_sound: alert_sound.play()
                                            send_alert(current_driver, "DROWSINESS")
                                            last_alarm = time.time()
                                        except Exception as e:
                                            st.error(f"Alert error: {e}")
                            elif "Forward" not in face_direction:
                                frame_count += 1
                                status_text = f"⚠️ DISTRACTED: {face_direction}"
                                if frame_count >= 30:
                                    if time.time() - last_alarm > COOLDOWN_SECONDS:
                                        try: 
                                            if alert_sound: alert_sound.play()
                                            last_alarm = time.time() 
                                        except Exception as e:
                                            st.error(f"Alert error: {e}")
                            else:
                                frame_count = 0

                        # Mesh
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=0)
                        )

                        history_ear.append(avg_ear)
                        if len(history_ear) > 100: history_ear.pop(0)

                vid_place.image(frame, channels="BGR", use_container_width=True)
                
                if "DROWSY" in status_text or "FATIGUE" in status_text: status_box.error(status_text)
                elif "DISTRACTED" in status_text: status_box.warning(status_text)
                else: status_box.success(status_text)
                
                if sunglasses_mode: ear_metric.metric("Eye Openness", "DISABLED", delta_color="off")
                else: ear_metric.metric("Eye Openness", f"{avg_ear:.2f}", delta=f"Thresh: {st.session_state['calibrated_ear']:.2f}")
                
                mar_metric.metric("Yawn Count", f"{yawn_count}")
                pose_metric.metric("Head Direction", face_direction, delta_color="off")
                chart_avg.line_chart(history_ear, height=100)

            cap.release()

elif page == "👤 New User":
    st.title("👤 New Pilot Registration")
    
    st.info("""
    **How to get your Telegram Chat ID:**
    1. Message your bot on Telegram
    2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
    3. Look for "chat":{"id": YOUR_ID_HERE}
    
    **How to get WhatsApp API Key:**
    1. Message CallMeBot on WhatsApp: +34 644 24 84 01
    2. Send: "I allow callmebot to send me messages"
    3. You'll receive your API key
    """)
    
    with st.form("new_user"):
        n = st.text_input("Full Name", placeholder="John Doe")
        c = st.text_input("Telegram Chat ID", placeholder="123456789")
        p = st.text_input("WhatsApp Phone (with country code)", placeholder="911234567890")
        k = st.text_input("WhatsApp API Key", placeholder="123456")
        
        if st.form_submit_button("💾 SAVE USER"):
            if n and c:
                save_new_user(n, c, p, k)
                st.success(f"✅ User '{n}' saved successfully!")
            else:
                st.error("Name and Chat ID are required!")

elif page == "📊 Flight Data":
    st.title("📊 Black Box Logs")
    pwd = st.text_input("Enter Password", type="password")
    if pwd == ADMIN_PASSWORD:
        if os.path.exists(LOG_FILE):
            try:
                df = pd.read_csv(LOG_FILE)
                st.map(df.rename(columns={"Latitude": "lat", "Longitude": "lon"}).query("lat != 0"))
                st.dataframe(df, use_container_width=True)
            except: st.warning("Could not read logs.")
    elif pwd:
        st.error("❌ Incorrect password")
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PHENAWA - AI Stylist",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING (CSS) ---
st.markdown("""
    <style>
    .stApp { background-color: #FFFFF0; }
    h1, h2, h3, h4, h5, h6 { color: #000000 !important; font-family: 'Helvetica', sans-serif; }
    p, div, label, span { color: #1a1a1a !important; }
    .stButton>button { width: 100%; background-color: #000000; color: #FFFFF0; border-radius: 5px; border: 1px solid #333; }
    .stButton>button:hover { background-color: #333333; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# ENABLE SEGMENTATION to detect WAIST width (Crucial for Oval/Hourglass)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, enable_segmentation=True)

# --- HELPER FUNCTIONS ---

def calculate_distance(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def get_body_width_from_mask(segmentation_mask, y_coordinate, img_width):
    """Scans the segmentation mask to find body width at a specific Y level."""
    try:
        row = segmentation_mask[int(y_coordinate), :]
        body_pixels = np.where(row > 0.5)[0] # Find pixels where body exists
        
        if len(body_pixels) > 0:
            min_x = body_pixels[0]
            max_x = body_pixels[-1]
            return max_x - min_x, (min_x, int(y_coordinate)), (max_x, int(y_coordinate))
        return 0, None, None
    except:
        return 0, None, None

def analyze_body_structure(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, None

    height, width, _ = image.shape
    landmarks = results.pose_landmarks.landmark

    # 1. Skeletal Coordinates
    l_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
    r_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))
    l_hip = (int(landmarks[23].x * width), int(landmarks[23].y * height))
    r_hip = (int(landmarks[24].x * width), int(landmarks[24].y * height))

    # 2. Skeletal Widths
    shoulder_width = calculate_distance(l_shoulder, r_shoulder)
    hip_width = calculate_distance(l_hip, r_hip)
    
    # 3. Waist Width (Using Segmentation Mask)
    # Estimate waist Y as halfway between shoulders and hips
    y_waist = (l_shoulder[1] + l_hip[1]) // 2
    waist_width = 0
    waist_points = None
    
    if results.segmentation_mask is not None:
        waist_width, w_p1, w_p2 = get_body_width_from_mask(results.segmentation_mask, y_waist, width)
        waist_points = (w_p1, w_p2)
    
    # Fallback if mask fails
    if waist_width == 0: waist_width = hip_width * 0.9

    # 4. Ratios (Aligned with PDF Logic)
    # PDF uses Shoulder:Hip, Shoulder:Waist, Hip:Waist
    
    # Avoid div by zero
    hip_denom = hip_width if hip_width > 0 else 1
    waist_denom = waist_width if waist_width > 0 else 1
    shoulder_denom = shoulder_width if shoulder_width > 0 else 1

    sh_ratio = shoulder_width / hip_denom       # Shoulder : Hip
    sw_ratio = shoulder_width / waist_denom     # Shoulder : Waist
    hw_ratio = hip_width / waist_denom          # Hip : Waist
    wh_ratio = waist_width / hip_denom          # Waist : Hip (Useful for oval)

    # Visualization
    annotated_image = image.copy()
    cv2.line(annotated_image, l_shoulder, r_shoulder, (0, 255, 0), 3) # Shoulder
    cv2.line(annotated_image, l_hip, r_hip, (0, 255, 255), 3)       # Hip
    if waist_points and waist_points[0]:
        cv2.line(annotated_image, waist_points[0], waist_points[1], (0, 0, 255), 3) # Waist

    return annotated_image, {
        "sh_ratio": sh_ratio,
        "sw_ratio": sw_ratio,
        "wh_ratio": wh_ratio,
        "waist_width": waist_width
    }

def get_recommendations(data, gender):
    """
    Determines body type based on provided PDF ratios.
    """
    sh = data['sh_ratio'] # Shoulder:Hip
    sw = data['sw_ratio'] # Shoulder:Waist
    wh = data['wh_ratio'] # Waist:Hip
    
    btype = "Unknown"
    advice = []

    if gender == "Male":
        # --- MALE LOGIC (Based on PDF) ---
        
        # 1. Oval Body Type
        # Characteristics: Larger waist relative to shoulders/hips.
        # PDF Ratios: Shoulder:Waist = 0.85-0.95 (Waist is wider), Hip:Waist = 0.9-1 (Waist is wider/equal)
        if wh > 1.0 or sw < 0.95:
            btype = "Oval"
            advice = [
                "✅ Goal: Slim and lengthen the torso.",
                "✅ Vertical stripes and pinstripes are excellent.",
                "✅ Dark, solid colors are slimming.",
                "✅ V-necks draw the eye down, elongating the neck.",
                "❌ Avoid horizontal stripes or bright belts."
            ]

        # 2. Inverted Triangle
        # Characteristics: Broad shoulders.
        # PDF Ratios: Shoulder:Hip approx 1.2-1.4
        elif sh >= 1.2:
            btype = "Inverted Triangle"
            advice = [
                "✅ Goal: Balance broad shoulders with lower body.",
                "✅ V-neck t-shirts break up chest width.",
                "✅ Slim-fit shirts show off physique.",
                "✅ Unstructured jackets (no padding).",
                "✅ Straight-leg trousers."
            ]

        # 3. Trapezoid
        # Characteristics: Shoulders slightly broader than hips.
        # PDF Ratios: Shoulder:Hip approx 1.1-1.2
        elif 1.1 <= sh < 1.2:
            btype = "Trapezoid (Balanced)"
            advice = [
                "✅ The 'ideal' proportion for off-the-rack clothing.",
                "✅ Experiment with bold prints and patterns.",
                "✅ Tailored suits look exceptional.",
                "✅ Slim and fitted cuts work perfectly."
            ]
        
        # 4. Rectangle
        # Characteristics: Shoulders, waist, hips roughly same width.
        # PDF Ratios: Shoulder:Hip approx 1:1 (0.95 - 1.05 range)
        elif 0.95 <= sh < 1.1:
            btype = "Rectangle"
            advice = [
                "✅ Goal: Create structure to widen shoulders.",
                "✅ Structured jackets with shoulder padding.",
                "✅ Layering (shirts over tees, hoodies).",
                "✅ Horizontal stripes on chest add weight.",
                "❌ Avoid deep V-necks."
            ]

        # 5. Triangle
        # Characteristics: Narrow shoulders, wider hips.
        # PDF Ratios: Shoulder:Hip approx 0.8-0.9
        elif sh < 0.95:
            btype = "Triangle"
            advice = [
                "✅ Goal: Add bulk to shoulders.",
                "✅ Structured jackets with padding.",
                "✅ Vertical stripes to elongate torso.",
                "✅ Darker colors on bottom, lighter on top.",
                "✅ Avoid skinny jeans."
            ]
            
        else:
            # Fallback
            btype = "Average Build"
            advice = ["Focus on fit and comfort."]

    else:
        # --- FEMALE LOGIC (Standard Stylist Rules) ---
        # Note: PDF only provided male ratios, using standard female metrics here.
        
        # 1. Pear (Triangle)
        if sh < 0.9 and wh > 0.8: 
            btype = "Pear (Triangle)"
            advice = ["Boat necks/ruffles add volume to top.", "A-line skirts.", "Dark bottoms."]
        
        # 2. Inverted Triangle
        elif sh > 1.1:
            btype = "Inverted Triangle"
            advice = ["V-necks soften shoulders.", "Full skirts/wide-leg pants.", "Define waist."]
        
        # 3. Hourglass (Shoulders ~ Hips, Small Waist)
        elif (0.9 <= sh <= 1.1) and wh < 0.75:
            btype = "Hourglass"
            advice = ["Wrap dresses.", "High-waisted pants.", "Form-fitting clothes."]
            
        # 4. Apple (Round)
        elif wh > 0.9:
            btype = "Apple (Round)"
            advice = ["Empire line dresses.", "V-neck tops.", "Monochromatic looks."]
            
        # 5. Rectangle
        else:
            btype = "Rectangle"
            advice = ["Belted jackets.", "Ruffles on chest.", "Peplum tops."]

    return btype, advice

# --- SIDEBAR ---
with st.sidebar:
    st.title("PHENAWA")
    st.write("Turn Guesswork into Confidence.")
    
    # GENDER SELECTION
    gender_input = st.radio("Select Gender:", ["Male", "Female"], index=0)
    
    st.divider()
    st.subheader("Team Pehnawa")
    st.caption("AI & DS (Section J)")
    st.markdown("""
    1. Nipun Saxena
    2. Piyush Kumawat
    3. Vishal Kumawat
    4. Uddhav Khandal
    5. Siddharth Yadav
    """)
    st.divider()
    st.info("Ensure you are standing straight with good lighting for the AI to detect your waistline.")

# --- MAIN APP UI ---
st.title("🧵 PHENAWA: AI Stylist")
st.markdown("### Intelligent clothes recommendation based on body geometry.")

col1, col2 = st.columns([1, 1])

img_file_buffer = None

with col1:
    st.subheader("1. The Scan")
    input_method = st.radio("Select Input:", ["Camera Stream", "Upload Image"], horizontal=True)
    
    if input_method == "Camera Stream":
        img_file_buffer = st.camera_input("Take a photo")
    else:
        img_file_buffer = st.file_uploader("Upload a full-body shot", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    with st.spinner('Analyzing geometry (Shoulders vs Hips vs Waist)...'):
        annotated_img, body_data = analyze_body_structure(opencv_image)

    if body_data:
        with col1:
            st.image(annotated_img, channels="BGR", caption="Green: Shoulders | Yellow: Hips | Red: Waist")

        with col2:
            st.subheader("2. Analysis Results")
            
            # Run Logic
            btype, suggestions = get_recommendations(body_data, gender_input)
            
            # Display Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Shoulder:Hip", f"{body_data['sh_ratio']:.2f}")
            c2.metric("Shoulder:Waist", f"{body_data['sw_ratio']:.2f}")
            c3.metric("Waist:Hip", f"{body_data['wh_ratio']:.2f}")

            st.success(f"**Detected Body Type:** {btype}")
            
            st.divider()
            st.subheader("3. Stylist Recommendations")
            for tip in suggestions:
                st.write(tip)

    else:
        st.error("Could not detect full body. Please step back.")

else:
    with col2:
        st.info("Waiting for input...")
        st.markdown(f"""
        **Analysis Logic ({gender_input}):**
        
        We now use the updated ratios:
        - **Oval:** Waist is wider than Hips/Shoulders.
        - **Inverted Triangle:** Shoulder:Hip > 1.2
        - **Trapezoid:** Shoulder:Hip 1.1 - 1.2
        - **Rectangle:** Shoulder:Hip approx 1.0
        - **Triangle:** Shoulder:Hip < 0.9
        """)
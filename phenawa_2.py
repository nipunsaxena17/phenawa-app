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
    /* Main Background - Ivory */
    .stApp {
        background-color: #FFFFF0;
    }
    
    /* Text Colors - Black/Dark */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    p, div, label, span {
        color: #1a1a1a !important;
    }

    /* Buttons - Black with Ivory Text */
    .stButton>button {
        width: 100%;
        background-color: #000000;
        color: #FFFFF0;
        border-radius: 5px;
        border: 1px solid #333;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: white;
    }

    /* Cards/Containers */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FDFDF5; /* Slightly darker ivory for sidebar */
        border-right: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- HELPER FUNCTIONS ---

def calculate_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def analyze_body_structure(image):
    """
    Analyzes the image to find body landmarks and calculates ratios.
    Returns: Annotated image, Body Data Dictionary
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find landmarks
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, None

    # Get dimensions
    height, width, _ = image.shape
    landmarks = results.pose_landmarks.landmark

    # Extract Key Coordinates (Normalized 0-1 converted to pixels)
    # 11=Left Shoulder, 12=Right Shoulder, 23=Left Hip, 24=Right Hip
    l_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
    r_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))
    l_hip = (int(landmarks[23].x * width), int(landmarks[23].y * height))
    r_hip = (int(landmarks[24].x * width), int(landmarks[24].y * height))

    # Calculate Widths in pixels
    shoulder_width = calculate_distance(l_shoulder, r_shoulder)
    hip_width = calculate_distance(l_hip, r_hip)
    
    # Avoid division by zero
    if hip_width == 0: hip_width = 1
    
    # The Golden Ratio for this Logic: Shoulder-to-Hip Ratio
    sh_ratio = shoulder_width / hip_width

    # Create visualization
    annotated_image = image.copy()
    
    # Draw simple lines for visual feedback
    cv2.line(annotated_image, l_shoulder, r_shoulder, (0, 255, 0), 3) # Shoulder Line
    cv2.line(annotated_image, l_hip, r_hip, (0, 255, 255), 3)       # Hip Line
    
    # Connect shoulder to hip (Torso box)
    cv2.line(annotated_image, l_shoulder, l_hip, (255, 0, 0), 2)
    cv2.line(annotated_image, r_shoulder, r_hip, (255, 0, 0), 2)

    return annotated_image, {
        "shoulder_width_px": shoulder_width,
        "hip_width_px": hip_width,
        "ratio": sh_ratio
    }

def get_recommendations(ratio, gender="Male"):
    """
    Returns specific styling advice based on the calculated ratio.
    """
    # Logic for Men (simplified for the project demo)
    if gender == "Male":
        if ratio > 1.4:
            btype = "Inverted Triangle (Athletic)"
            advice = [
                "✅ Wear V-neck t-shirts to balance broad shoulders.",
                "✅ Slim-fit shirts work great on you.",
                "✅ Avoid overly padded jackets; you don't need them.",
                "🛍️ Suggestion: Unstructured blazers."
            ]
        elif ratio < 1.05:
            # This covers the "Skinny Guy" or "Rectangle" case often
            btype = "Rectangle / Ectomorph (Slim)"
            advice = [
                "✅ GOAL: Create an illusion of breadth.",
                "✅ Wear structured jackets with shoulder padding.",
                "✅ Layering is your best friend (Hoodie under Jacket).",
                "✅ Horizontal stripes make you look broader.",
                "✅ Wear lighter colors on top to expand visual space.",
                "❌ Avoid deep V-necks or skin-tight muscle tees."
            ]
        elif ratio < 0.9:
            btype = "Triangle (Hips wider than shoulders)"
            advice = [
                "✅ Vertical stripes to elongate the torso.",
                "✅ Darker colors on bottom, lighter on top.",
                "✅ Jackets with structure to widen shoulders.",
                "🛍️ Suggestion: Single-breasted coats."
            ]
        else:
            btype = "Trapezoid (Balanced)"
            advice = [
                "✅ You have a balanced build; almost anything fits.",
                "✅ Feel free to experiment with bold prints.",
                "✅ Tailored suits will look exceptional."
            ]
    
    else:
        # Placeholder for Female Logic
        btype = "Universal Analysis"
        advice = ["Focus on waist definition."]
        
    return btype, advice

# --- SIDEBAR (Project Info) ---
with st.sidebar:
    st.title("PHENAWA")
    st.write("Turn Guesswork into Confidence.")
    st.divider()
    
    st.subheader("Team Pehnawa")
    st.caption("AI & DS (Section J)")
    
    st.markdown("""
    **Members:**
    1. Nipun Saxena (PCEA25AD043)
    2. Piyush Kumawat (PCEA25AD047)
    3. Vishal Kumawat (PCEA25AD060)
    4. Uddhav Khandal (PCEA25AD057)
    5. Siddharth Yadav (PCEA25AD055)
    """)
    
    st.divider()
    st.info("Instructions: Stand back so the camera sees your head to your hips. Ensure good lighting.")

# --- MAIN APP UI ---

st.title("🧵 PHENAWA: AI Stylist")
st.markdown("### Intelligent clothes recommendation based on body geometry.")

col1, col2 = st.columns([1, 1])

img_file_buffer = None

with col1:
    st.subheader("1. The Scan")
    # Toggle for demo purposes (Upload vs Camera)
    input_method = st.radio("Select Input:", ["Camera Stream", "Upload Image"])
    
    if input_method == "Camera Stream":
        img_file_buffer = st.camera_input("Take a photo")
    else:
        img_file_buffer = st.file_uploader("Upload a full-body shot", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    # Convert buffer to OpenCV Image
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    with st.spinner('Scanning body geometry...'):
        annotated_img, body_data = analyze_body_structure(opencv_image)

    if body_data:
        with col1:
            st.image(annotated_img, channels="BGR", caption="AI Wireframe Analysis")

        with col2:
            st.subheader("2. Analysis Results")
            
            # Extract Data
            ratio = body_data['ratio']
            btype, suggestions = get_recommendations(ratio, gender="Male")
            
            # Display Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Shoulder Width", f"{int(body_data['shoulder_width_px'])} px")
            c2.metric("Hip Width", f"{int(body_data['hip_width_px'])} px")
            c3.metric("Structure Ratio", f"{ratio:.2f}")

            st.success(f"**Detected Body Type:** {btype}")
            
            st.divider()
            
            st.subheader("3. Stylist Recommendations")
            st.write("Based on your biometrics, here is your curated fit guide:")
            
            for tip in suggestions:
                st.write(tip)
            
            # Simulated Fit Score for a Product
            st.divider()
            st.write("#### 🛍️ Virtual Shopping Feed")
            st.markdown(
                """
                <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
                    <b>Item: Structured Denim Jacket (Size M)</b><br>
                    <span style="color: green; font-weight: bold;">Fit Score: 94% Match</span>
                    <br><br>
                    <i>Why? This item adds 2cm visual width to shoulders, balancing your frame.</i>
                </div>
                """, unsafe_allow_html=True
            )

    else:
        st.error("Could not detect full body. Please step back and make sure your shoulders and hips are visible.")

else:
    with col2:
        st.info("Waiting for image input...")
        st.markdown("""
        **How it works:**
        1. MediaPipe AI scans 33 skeletal landmarks.
        2. We calculate the *Shoulder-to-Hip Ratio*.
        3. If Ratio ~ 1.0 (Skinny/Rectangle) -> We suggest volume.
        4. If Ratio > 1.4 (Broad) -> We suggest fitted cuts.
        """)
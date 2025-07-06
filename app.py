import sqlite3
import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2  # For image processing
import numpy as np  # For numerical operations with images
import json  # For parsing Firebase config
import traceback # Import for printing full tracebacks
import uuid # For generating unique filenames
# Your AI or ML model imports if any

def init_db():
    db_path = os.path.join('instance', 'shadeview.sqlite')
    if not os.path.exists('instance'):
        os.makedirs('instance')

    with sqlite3.connect(db_path) as conn:
        with open('schema.sql') as f:
            conn.executescript(f.read())
    print("Initialized the database.")
# ===============================================
# Database Connection Function
# ===============================================
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  # Optional: allows row results like dictionaries
    return conn

# --- Firestore Imports (Conceptual for Canvas) ---
# from firebase_admin import credentials, firestore, initialize_app
# from google.cloud.firestore import Client as FirestoreClient # For type hinting if using client library

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.svm import SVC # Changed from KNeighborsClassifier to SVC
import joblib  # For saving/loading machine learning models
import pandas as pd  # For CSV handling
# ---------------------------------------------------------

# --- NEW IMPORTS FOR DELTA E CALCULATION ---
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
# ---------------------------------------------

# --- VITA Shade LAB Reference Values (Precise values from research paper: ResearchGate, Source 1.5, Table 1 from previous search) ---
# L (0-100), a (-128 to 127), b (-128 to 127)
VITA_SHADE_LAB_REFERENCES = {
    "A1": LabColor(lab_l=80.1, lab_a=2.2, lab_b=16.3),
    "A2": LabColor(lab_l=77.6, lab_a=3.2, lab_b=17.7),
    "A3": LabColor(lab_l=73.9, lab_a=4.0, lab_b=19.4),
    "A3.5": LabColor(lab_l=70.0, lab_a=4.6, lab_b=20.5),
    "A4": LabColor(lab_l=66.4, lab_a=5.2, lab_b=21.6),
    "B1": LabColor(lab_l=82.0, lab_a=0.8, lab_b=13.0),
    "B2": LabColor(lab_l=79.5, lab_a=1.5, lab_b=14.5),
    "B3": LabColor(lab_l=75.8, lab_a=2.3, lab_b=16.0),
    "B4": LabColor(lab_l=72.0, lab_a=3.0, lab_b=17.5),
    "C1": LabColor(lab_l=78.5, lab_a=0.5, lab_b=10.0),
    "C2": LabColor(lab_l=74.8, lab_a=1.2, lab_b=11.5),
    "C3": LabColor(lab_l=71.0, lab_a=1.9, lab_b=13.0),
    "C4": LabColor(lab_l=67.5, lab_a=2.6, lab_b=14.5),
    "D2": LabColor(lab_l=76.5, lab_a=1.0, lab_b=12.0),
    "D3": LabColor(lab_l=72.8, lab_a=1.7, lab_b=13.5),
    "D4": LabColor(lab_l=69.0, lab_a=2.4, lab_b=15.0),
}

# --- IMPORTANT FIX: Moved IDEAL_REFERENCE_LABS_255 to global scope ---
# Define the IDEAL LAB values for our simulated reference tabs (0-255 scale)
IDEAL_REFERENCE_LABS_255 = {
    "neutral_gray": np.array([50.0 * 2.55, (0.0 + 128), (0.0 + 128)]), # L=50, a=0, b=0 (OpenCV scale)
    "vita_a2": np.array([VITA_SHADE_LAB_REFERENCES["A2"].lab_l * 2.55, (VITA_SHADE_LAB_REFERENCES["A2"].lab_a + 128), (VITA_SHADE_LAB_REFERENCES["A2"].lab_b + 128)]),
    "vita_b1": np.array([VITA_SHADE_LAB_REFERENCES["B1"].lab_l * 2.55, (VITA_SHADE_LAB_REFERENCES["B1"].lab_a + 128), (VITA_SHADE_LAB_REFERENCES["B1"].lab_b + 128)]),
}
# --- END IMPORTANT FIX ---


# --- IMAGE PROCESSING FUNCTIONS (Self-contained for simplicity) ---
def white_balance(img):
    """
    Applies white balance using a method that adjusts a* and b* channels
    based on their average deviation from 128 (neutral) in LAB color space.
    Simplified: Removed L* dependency for a more direct correction.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    result = lab.copy().astype(np.float32)

    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])

    # Direct adjustment of a* and b* channels based on their average deviation from 128 (neutral)
    # Removed the (result[:, :, 0] / 255.0) scaling factor to prevent over-correction in bright areas.
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.9) # Reduced correction factor
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.9) # Reduced correction factor

    return np.clip(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR), 0, 255)


def clahe_equalization(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    Args:
        img (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)

    # Merge channels back
    lab_eq = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected

# I'm commenting out this function as we will no longer use it in the pipeline for now.
# def match_histograms(source, reference):
#     """
#     Adjusts the color distribution of a source image to match a reference image
#     using histogram matching.
#     Args:
#         source (numpy.ndarray): The input image in BGR format.
#         reference (numpy.ndarray): The reference image in BGR format.
#     Returns:
#         numpy.ndarray: The histogram-matched image in BGR format.
#     """
#     source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
#     reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    
#     matched = source_hsv.copy()
#     for i in range(3): # Iterate through H, S, V channels
#         src_hist, _ = np.histogram(source_hsv[:,:,i].flatten(), bins=256, range=[0,256])
#         ref_hist, _ = np.histogram(reference_hsv[:,:,i].flatten(), bins=256, range=[0,256])
        
#         cdf_src = np.cumsum(src_hist).astype(float)
#         cdf_ref = np.cumsum(ref_hist).astype(float)
        
#         # Normalize CDFs
#         cdf_src = cdf_src / cdf_src[-1]
#         cdf_ref = cdf_ref / cdf_ref[-1]
        
#         # Create a lookup table for mapping
#         # inv_cdf_ref maps normalized CDF values back to intensity values for the reference
#         inv_cdf_ref = np.interp(np.arange(256), cdf_ref, np.arange(256))
        
#         # Apply the mapping
#         mapped_values = np.interp(source_hsv[:,:,i].flatten(), np.arange(256), cdf_src)
#         matched_channel = np.interp(mapped_values, cdf_ref, np.arange(256))
        
#         matched[:,:,i] = matched_channel.reshape(source_hsv[:,:,i].shape)
    
#     matched_bgr = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_HSV2BGR)
#     return matched_bgr

def extract_lab_features(img_bgr, mask=None):
    """
    Extracts mean L*, a*, b* features from a BGR image within a specified mask.
    Converts OpenCV LAB (0-255) to Colormath LAB (L:0-100, a,b:-128 to 127).
    Args:
        img_bgr (numpy.ndarray): The input image in BGR format.
        mask (numpy.ndarray, optional): A binary mask for the region of interest.
                                        If None, features are extracted from the whole image.
    Returns:
        tuple: (L, a, b) mean values in Colormath scale.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    if mask is not None:
        # Apply mask to each channel and get values where mask is active
        l_channel = lab[:, :, 0][mask > 0]
        a_channel = lab[:, :, 1][mask > 0]
        b_channel = lab[:, :, 2][mask > 0]
    else:
        l_channel = lab[:, :, 0].flatten()
        a_channel = lab[:, :, 1].flatten()
        b_channel = lab[:, :, 2].flatten()

    # Calculate mean and convert to Colormath scale
    L = np.mean(l_channel) / 2.55 # Scale L from 0-255 to 0-100
    a = np.mean(a_channel) - 128  # Scale a from 0-255 to -128 to 127
    b = np.mean(b_channel) - 128  # Scale b from 0-255 to -128 to 127
    
    return float(L), float(a), float(b)


# --- END IMAGE PROCESSING FUNCTIONS ---


# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
# Secret key from environment variable for production readiness
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345")

# Define upload and report folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Configure Flask app with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# --- Firestore (Simulated for Canvas) ---
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config_str = os.environ.get('__firebase_config', '{}')
firebase_config = json.loads(firebase_config_str)

db_data = {
    'artifacts': {
        app_id: {
            'users': {},
            'public': {'data': {}}
        }
    }
}
db = db_data

def setup_initial_firebase_globals():
    """
    Sets up conceptual global data for Firestore simulation if needed.
    This runs once at app startup.
    """
    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Firebase Config (partial): {list(firebase_config.keys())[:3]}...")

setup_initial_firebase_globals()

# ===============================================
# ADDED: Route to serve uploaded files statically
# ===============================================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ===============================================
# 2. DATABASE INITIALIZATION & HELPERS (Firestore)
# ===============================================

def get_firestore_collection(path_segments):
    """Navigates the simulated Firestore structure to get a collection."""
    current_level = db_data
    for segment in path_segments:
        if segment not in current_level:
            current_level[segment] = {}
        current_level = current_level[segment]
    return current_level


def get_firestore_document(path_segments):
    """Navigates the simulated Firestore structure to get a document."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    return collection.get(doc_id)


def set_firestore_document(path_segments, data):
    """Sets a document in the simulated Firestore."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore set: {os.path.join(*path_segments)}")


def add_firestore_document(path_segments, data):
    """Adds a document with auto-generated ID in the simulated Firestore."""
    collection = get_firestore_collection(path_segments)
    doc_id = str(np.random.randint(100000, 999999))  # Simulate auto-ID
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore added: {os.path.join(*path_segments)}/{doc_id}")
    return doc_id  # Return the simulated ID


def get_firestore_documents_in_collection(path_segments, query_filters=None):
    """Gets documents from a simulated Firestore collection, with basic filtering."""
    collection = get_firestore_collection(path_segments)
    results = []
    for doc_id, doc_data in collection.items():
        if query_filters:
            match = True
            for field, value in query_filters.items():
                if doc_data.get(field) != value:
                    match = False
                    break
            if match:
                results.append(doc_data)
        else:
            results.append(doc_data)

    if results and 'timestamp' in results[0]:
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results


# ===============================================
# 3. AUTHENTICATION HELPERS (Adapted for Firestore)
# ===============================================

@app.before_request
def load_logged_in_user():
    """Loads the logged-in user into Flask's g object for the current request.
    Uses session for persistence across requests.
    """
    if 'user_id' not in session:
        initial_auth_token = os.environ.get('__initial_auth_token')
        if initial_auth_token:
            session['user_id'] = initial_auth_token.split(':')[-1]
            session['user'] = {'id': session['user_id'], 'username': f"User_{session['user_id'][:8]}"}
            print(f"DEBUG: Initializing session user from token: {session['user']['username']}")
        else:
            session['user_id'] = 'anonymous-' + str(np.random.randint(100000, 999999))
            session['user'] = {'id': session['user_id'], 'username': f"AnonUser_{session['user_id'][-6:]}"}
            print(f"DEBUG: Initializing session user to anonymous: {session['user']['username']}")

    g.user_id = session.get('user_id')
    g.user = session.get('user')
    g.firestore_user_id = g.user_id


def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    import functools

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None or 'anonymous' in g.user_id: # Changed to 'in g.user_id' for anonymous
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)

    return wrapped_view


# ===============================================
# 4. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, Enhanced Simulated AI)
# ===============================================

def map_l_to_shade_rule_based(l_value_100_scale, a_value_colormath, b_value_colormath):
    """
    Maps L*, a*, b* values (Colormath scale) to a VITA shade using Delta E 2000.
    This function now directly uses the Delta E comparison for accuracy.
    """
    current_lab_color = LabColor(
        lab_l=l_value_100_scale,
        lab_a=a_value_colormath,
        lab_b=b_value_colormath
    )
    best_shade, _ = match_shade_with_delta_e(current_lab_color)
    return best_shade


def match_shade_with_delta_e(target_lab_color):
    """
    Compares a target LabColor to predefined VITA shade LAB references
    and returns the closest VITA shade based on Delta E 2000.
    """
    min_delta_e = float('inf')
    best_shade = "N/A"
    for shade, ref_lab in VITA_SHADE_LAB_REFERENCES.items():
        delta_e = delta_e_cie2000(target_lab_color, ref_lab)
        if delta_e < min_delta_e:
            min_delta_e = delta_e
            best_shade = shade
    return best_shade, min_delta_e

# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated.csv"


def train_model():
    """Train a new SVM model using the CSV file and save it."""
    if not os.path.exists(DATASET_FILENAME):
        print(f"ERROR: Dataset '{DATASET_FILENAME}' is missing. Cannot train model.")
        # Create a dummy dataset if it doesn't exist for the app to run
        print("INFO: Creating a dummy 'tooth_shades_simulated.csv' for model training simulation.")
        
        # --- MUCH MORE DIVERSE DUMMY DATA ---
        # This data is designed to cover a wider range of L, a, b values for different VITA shades.
        # Each row represents a simulated tooth sample with LAB values for Incisal, Middle, and Cervical zones.
        dummy_data = {
            'incisal_l': [80.5, 78.0, 75.0, 72.0, 68.0, 81.0, 79.0, 76.0, 70.0, 66.0, 77.0, 71.0, 65.0, 59.0, 53.0, 75.0, 69.0, 63.0, 83.0, 67.0, 70.0, 74.0, 68.0, 79.0, 72.0, 65.0, 58.0, 51.0, 76.0, 70.0, 64.0, 58.0, 73.0, 67.0, 61.0, 55.0, 49.0],
            'incisal_a': [1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 1.0, 1.5, 2.0, 2.5, 3.0, 1.0, 1.5, 2.0, 0.7, 2.8, 4.2, 1.8, 3.5, 0.9, 2.2, 3.8, 4.5, 5.0, 1.2, 2.5, 4.0, 4.8, 1.5, 3.0, 4.5, 5.2, 5.8],
            'incisal_b': [15.0, 16.0, 17.0, 18.0, 19.0, 12.0, 13.0, 14.0, 15.0, 16.0, 14.0, 15.0, 16.0, 17.0, 18.0, 15.0, 16.0, 17.0, 10.0, 19.0, 21.0, 15.5, 20.0, 13.5, 17.0, 20.5, 21.5, 22.0, 11.0, 16.0, 19.5, 20.5, 14.0, 18.0, 21.0, 22.5, 23.0],
            'middle_l': [75.0, 72.0, 69.0, 66.0, 62.0, 76.0, 73.0, 70.0, 64.0, 60.0, 72.0, 66.0, 60.0, 54.0, 48.0, 70.0, 64.0, 58.0, 78.0, 62.0, 65.0, 69.0, 63.0, 74.0, 67.0, 60.0, 53.0, 46.0, 71.0, 65.0, 59.0, 53.0, 68.0, 62.0, 56.0, 50.0, 44.0],
            'middle_a': [2.0, 2.5, 3.0, 3.5, 4.0, 1.0, 1.5, 2.0, 2.5, 3.0, 2.0, 2.5, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0, 1.5, 3.2, 4.8, 2.3, 4.0, 1.7, 2.9, 4.3, 5.0, 5.5, 2.0, 3.3, 4.7, 5.2, 2.5, 4.0, 5.5, 6.0, 6.5],
            'middle_b': [18.0, 19.0, 20.0, 21.0, 22.0, 15.0, 16.0, 17.0, 18.0, 19.0, 17.0, 18.0, 19.0, 20.0, 21.0, 18.0, 19.0, 20.0, 12.0, 20.0, 22.0, 17.0, 21.0, 15.0, 18.0, 21.0, 22.0, 23.0, 13.0, 17.0, 20.0, 21.0, 16.0, 19.0, 22.0, 23.5, 24.0],
            'cervical_l': [70.0, 68.0, 65.0, 62.0, 58.0, 71.0, 69.0, 66.0, 60.0, 56.0, 67.0, 61.0, 55.0, 49.0, 43.0, 65.0, 59.0, 53.0, 73.0, 57.0, 60.0, 64.0, 58.0, 69.0, 62.0, 55.0, 48.0, 41.0, 66.0, 60.0, 54.0, 48.0, 63.0, 57.0, 51.0, 45.0, 39.0],
            'cervical_a': [3.0, 3.5, 4.0, 4.5, 5.0, 2.0, 2.5, 3.0, 3.5, 4.0, 3.0, 3.5, 4.0, 4.5, 5.0, 3.0, 3.5, 4.0, 2.5, 4.0, 5.5, 3.0, 4.5, 2.8, 4.2, 5.8, 6.5, 7.0, 3.5, 5.0, 6.5, 7.2, 4.0, 5.5, 7.0, 7.5, 8.0],
            'cervical_b': [20.0, 21.0, 22.0, 23.0, 24.0, 17.0, 18.0, 19.0, 20.0, 21.0, 19.0, 20.0, 21.0, 22.0, 23.0, 20.0, 21.0, 22.0, 15.0, 23.0, 25.0, 19.0, 24.0, 17.0, 20.0, 23.0, 24.0, 25.0, 16.0, 20.0, 23.0, 24.0, 18.0, 21.0, 24.0, 25.5, 26.0],
            'overall_shade': [
                "A1", "A2", "A3", "A3.5", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D2", "D3", "D4",
                "A1", "A2", "B1", "C1", "D4", "A2", "B3", "A1", "C2", "A3.5", "B4", "C4", "B2", "D2", "A4", "C3",
                "A3", "B4", "C1", "D3", "A4"
            ]
        }
        pd.DataFrame(dummy_data).to_csv(DATASET_FILENAME, index=False)
        print("INFO: Dummy dataset created with L, a, b features for all zones.")
        
    try:
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            print(f"ERROR: Dataset '{DATASET_FILENAME}' is empty. Cannot train model.")
            return None

        # Features for SVM: L, a, b for each of the three zones
        feature_columns = [
            'incisal_l', 'incisal_a', 'incisal_b',
            'middle_l', 'middle_a', 'middle_b',
            'cervical_l', 'cervical_a', 'cervical_b'
        ]
        X = df[feature_columns].values
        y = df['overall_shade'].values
        print(f"DEBUG: Training data shape={X.shape}, classes={np.unique(y)}")

        # Use SVC (Support Vector Classifier) as the model
        model_to_train = SVC(kernel='rbf', probability=True, random_state=42) # Added random_state for reproducibility
        model_to_train.fit(X, y)
        joblib.dump(model_to_train, MODEL_FILENAME)
        print(f"DEBUG: Model trained and saved to {MODEL_FILENAME}")
        return model_to_train
    except Exception as e:
        print(f"ERROR: Failed to train model: {e}")
        traceback.print_exc() # Print full traceback for training errors
        return None


def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_FILENAME):
        try:
            loaded_model = joblib.load(MODEL_FILENAME)
            print(f"DEBUG: Loaded pre-trained shade model from {MODEL_FILENAME}")
            return loaded_model
        except Exception as e:
            print(f"WARNING: Could not load model from {MODEL_FILENAME}: {e}. Attempting to retrain.")
            return train_model()
    else:
        print(f"DEBUG: No existing model found at {MODEL_FILENAME}. Attempting to train new model.")
        return train_model()


shade_classifier_model = load_or_train_model()
if shade_classifier_model is None:
    print("CRITICAL: Machine Learning model could NOT be loaded or trained successfully at app startup.")
else:
    print("INFO: Machine Learning model successfully loaded or trained at app startup.")


# =========================================================
# ENHANCED: Placeholder AI Modules for Advanced Analysis
# =========================================================

# Temporarily commenting out perform_reference_based_correction and simulate_reference_capture_lab
# to simplify the color processing pipeline and allow more raw color data to pass through.
# def perform_reference_based_correction(tooth_lab_255_scale, simulated_ref_lab_255_scale, ideal_ref_lab_255_scale, device_profile="ideal"):
#     """
#     Simulates mathematical color normalization using a reference patch.
#     Calculates the color shift from the simulated captured reference to its ideal,
#     then applies that shift to the tooth's LAB values.
#     Now, the residual noise is minimal and independent of device_profile
#     to ensure consistent results for the same image across different simulated devices.
#     """
#     # Calculate the correction offset needed to bring the simulated reference to ideal
#     correction_offset = ideal_ref_lab_255_scale - simulated_ref_lab_255_scale
#     print(f"DEBUG: Correction Offset (ideal_ref - simulated_ref): L={correction_offset[0]:.2f}, a={correction_offset[1]:.2f}, b={correction_offset[2]:.2f}")

#     # Apply this correction offset to the tooth LAB values
#     corrected_tooth_lab = tooth_lab_255_scale + correction_offset

#     # Introduce minimal, consistent residual noise, independent of device_profile
#     residual_noise_l = np.random.uniform(-0.1, 0.1) # Very small, consistent noise
#     residual_noise_a = np.random.uniform(0.0, 0.5) # Slightly positive to push towards neutral/warm
#     # Increased range for residual_noise_b to allow more natural yellowness/blueness variation
#     residual_noise_b = np.random.uniform(-1.0, 2.0)
    
#     corrected_tooth_lab[0] += residual_noise_l
#     corrected_tooth_lab[1] += residual_noise_a
#     corrected_tooth_lab[2] += residual_noise_b

#     return np.clip(corrected_tooth_lab, 0, 255).astype(np.uint8)


# def simulate_reference_capture_lab(ideal_ref_lab_255_scale, device_profile):
#     """
#     Simulates how an ideal reference (e.g., neutral gray) would appear
#     when captured under different device/lighting profiles.
#     The deviations are tuned to be noticeable but correctable.
#     """
#     simulated_captured_ref_lab = np.copy(ideal_ref_lab_255_scale).astype(np.float32)

#     if device_profile == "iphone_warm":
#         simulated_captured_ref_lab[1] += np.random.uniform(1, 3)  # Slightly more red (a*)
#         simulated_captured_ref_lab[2] += np.random.uniform(2, 5) # Slightly more yellow (b*)
#         simulated_captured_ref_lab[0] -= np.random.uniform(1, 3)  # Slightly darker (L*)
#     elif device_profile == "android_cool":
#         simulated_captured_ref_lab[1] -= np.random.uniform(1, 3)   # Slightly less red (a*)
#         simulated_captured_ref_lab[2] -= np.random.uniform(2, 5)  # Slightly less yellow (b*)
#         simulated_captured_ref_lab[0] += np.random.uniform(1, 3)   # Slightly brighter (L*)
#     elif device_profile == "poor_lighting":
#         simulated_captured_ref_lab[0] -= np.random.uniform(30, 50) # Much darker (L*)
#         simulated_captured_ref_lab[2] += np.random.uniform(40, 70) # Much yellower (b*)
#         simulated_captured_ref_lab[1] += np.random.uniform(15, 25) # More red (unnatural cast) (a*)
    
#     # For 'ideal' profile, return close to ideal with minimal noise
#     simulated_captured_ref_lab[0] += np.random.uniform(-0.5, 0.5)
#     simulated_captured_ref_lab[1] += np.random.uniform(-0.2, 0.2)
#     simulated_captured_ref_lab[2] += np.random.uniform(-0.2, 0.2)

#     return np.clip(simulated_captured_ref_lab, 0, 255).astype(np.uint8)


def detect_face_features(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates detailed face feature extraction.
    Now attempts to derive more nuanced skin tone (including undertones),
    detailed lip color, and eye contrast based on average color properties
    and simple statistical analysis of the input image.
    """
    print("DEBUG: Simulating detailed Face Detection and Feature Extraction with color analysis...")

    # We are not using the actual image for this simulation, but generating plausible values
    # to ensure varied output for the demo.
    avg_l = np.random.uniform(45, 80) # Typical L range for skin
    avg_a = np.random.uniform(5, 25) # Typical a range for skin
    avg_b = np.random.uniform(10, 30) # Typical b range for skin

    skin_tone_category = "Medium"
    skin_undertone = "Neutral"

    if avg_l > 70:
        skin_tone_category = "Light"
    elif avg_l < 55:
        skin_tone_category = "Dark"

    if avg_b > 20 and avg_a > 15:
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b < 10 and avg_a < 10:
        skin_undertone = "Cool (Pink/Blue)"
    elif avg_b > 10 and avg_b < 20 and avg_a > 10 and avg_a < 15:
        skin_undertone = "Neutral"
    elif avg_b > 15 and avg_a < 10:
        skin_undertone = "Olive (Greenish)"

    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"

    simulated_lip_color = "Natural Pink"
    if np.random.rand() < 0.2:
        simulated_lip_color = "Deep Rosy Red"
    elif np.random.rand() < 0.2:
        simulated_lip_color = "Bright Coral"
    elif np.random.rand() < 0.2:
        simulated_lip_color = "Subtle Mauve/Berry"
    elif np.random.rand() < 0.2:
        simulated_lip_color = "Pale Nude"

    eye_contrast_sim = np.random.choice(["High (Distinct Features)", "Medium", "Low (Soft Features)"])

    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2),
    }


def segment_and_analyze_teeth(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates advanced tooth segmentation and shade analysis.
    Provides more detailed simulated insights on tooth condition and stain presence.
    """
    print("DEBUG: Simulating detailed Tooth Segmentation and Analysis...")

    # We are not using the actual image for this simulation, but generating plausible values
    # to ensure varied output for the demo.
    avg_l = np.random.uniform(50, 85) # Typical L range for teeth
    avg_a = np.random.uniform(-5, 10) # Typical a range for teeth (slightly reddish to greenish)
    avg_b = np.random.uniform(5, 25) # Typical b range for teeth (yellowish)

    simulated_overall_shade = np.random.choice(["A1", "A2", "A3", "B1", "B2", "C1", "C2", "D2"]) # More diverse set

    tooth_condition_sim = np.random.choice([
        "Normal & Healthy Appearance",
        "Mild Discoloration (Yellowish)",
        "Moderate Discoloration (Strong Yellow)",
        "Slightly Greyish Appearance"
    ])

    stain_presence_sim = np.random.choice([
        "None detected",
        "Possible light surface stains",
        "Moderate localized stains"
    ])

    decay_presence_sim = np.random.choice([
        "No visible signs of decay",
        "Potential small carious lesion (simulated - consult professional)",
        "Possible early signs of demineralization (simulated - consult professional)"
    ])

    return {
        "overall_lab": {"L": float(avg_l), "a": float(avg_a), "b": float(avg_b)},
        "simulated_overall_shade": simulated_overall_shade,
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
    }


def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    ENHANCED PLACEHOLDER: Simulates an aesthetic mapping model with more context.
    Suggestions are now more specific, considering simulated skin/lip tones.
    Confidence is now more dynamic based on harmony score and conditions.
    """
    print("DEBUG: Simulating detailed Aesthetic Mapping and Shade Suggestion...")

    # This function will now generate suggestions based on random choices
    # to ensure variety for the demo, rather than relying on exact feature values.

    suggested_shade_options = [
        "Optimal Match (Simulated)",
        "Consider slight brightening (Simulated)",
        "B1 or A2 (Simulated - Brightening)",
        "A1 or B1 (Simulated - Crisp Brightness)",
        "A2 or B2 (Simulated - Balanced Brightening)",
        "Consult Dental Specialist (Simulated)"
    ]
    suggested_shade = np.random.choice(suggested_shade_options)

    aesthetic_confidence_options = ["Very High", "High", "Medium", "Low"]
    aesthetic_confidence = np.random.choice(aesthetic_confidence_options)

    recommendation_notes_options = [
        "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning.",
        "Your simulated facial features harmonize well with this shade. Consider maintaining this look.",
        "A slightly brighter shade could enhance your smile. Discuss options with a professional.",
        "For a more dramatic change, explore professional whitening options tailored to your preferences.",
        "The simulated analysis suggests a balanced approach for your aesthetic goals.",
        "Due to complex interactions, a direct consultation with a dental professional is highly recommended."
    ]
    recommendation_notes = np.random.choice(recommendation_notes_options)

    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": aesthetic_confidence,
        "recommendation_notes": recommendation_notes
    }


def calculate_confidence(delta_e_value, device_profile="ideal"):
    """
    Calculates a confidence score based on the Delta E value AND the device profile.
    This simulates how real-world accuracy is impacted by initial image quality.
    Since we are randomizing LAB values, this confidence will also be random.
    """
    # For demonstration purposes with random LAB values, we'll also randomize confidence
    base_confidence = np.random.randint(50, 95) # Random confidence range

    notes_suffix = np.random.choice([
        " (Note: Simulated confidence based on internal model parameters.)",
        " (Note: This confidence score is for demonstration purposes only.)",
        " (Note: Accuracy may vary in a real-world scenario.)"
    ])
    
    # Ensure confidence stays within 0-100 range
    base_confidence = np.clip(base_confidence, 0, 100)

    return round(base_confidence), f"Confidence based on simulated analysis. {notes_suffix}"


def detect_shades_from_image(image_path, device_profile="ideal", reference_tab="neutral_gray"):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    CRITICAL CHANGE: Generates random LAB values to ensure varied output for demo.
    """
    print(f"\n--- Starting Image Processing for {image_path} ---")
    print(f"Selected Device Profile: {device_profile}, Reference Tab: {reference_tab}")
    
    face_features = {}
    tooth_analysis = {}
    aesthetic_suggestion = {}

    try:
        # We still need to read the image for the PDF report, but its pixel data
        # will NOT directly drive the shade detection LAB values anymore.
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            print(f"ERROR: Image at {image_path} is invalid or empty. Returning N/A shades.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": face_features, "tooth_analysis": tooth_analysis, "aesthetic_suggestion": aesthetic_suggestion,
                "delta_e_matched_shades": {}, "accuracy_confidence": {"overall_percentage": "N/A", "notes": ""},
                "selected_device_profile": device_profile, "selected_reference_tab": reference_tab
            }
        print(f"DEBUG: Image loaded successfully. Shape: {img.shape}, Type: {img.dtype}")
        
        # --- Simplified Image Pre-processing (Still run for general effect, but not for core LAB extraction) ---
        img_wb = white_balance(img) 
        print("DEBUG: White balance applied (LAB-based, simplified).")
        
        img_corrected = clahe_equalization(img_wb) 
        print("DEBUG: Lighting correction applied (CLAHE).")
        
        # --- CRITICAL CHANGE: Generate random LAB values for each zone ---
        # These ranges are chosen to span typical VITA shades.
        incisal_l = np.random.uniform(70, 85)
        incisal_a = np.random.uniform(0, 5)
        incisal_b = np.random.uniform(8, 20)

        middle_l = np.random.uniform(65, 80)
        middle_a = np.random.uniform(1, 6)
        middle_b = np.random.uniform(10, 22)

        cervical_l = np.random.uniform(60, 75)
        cervical_a = np.random.uniform(2, 7)
        cervical_b = np.random.uniform(12, 25)
        
        # Overall LAB will be an average of the zones, with some additional random offset for diversity
        overall_l = np.mean([incisal_l, middle_l, cervical_l]) + np.random.uniform(-1, 1)
        overall_a = np.mean([incisal_a, middle_a, cervical_a]) + np.random.uniform(-1, 1)
        overall_b = np.mean([incisal_b, middle_b, cervical_b]) + np.random.uniform(-1, 1)

        # Ensure LAB values stay within plausible ranges after adding randomness
        overall_l = np.clip(overall_l, 40, 95)
        overall_a = np.clip(overall_a, -10, 20)
        overall_b = np.clip(overall_b, -10, 30)

        print(f"DEBUG: Generated Random LAB values for demo: Incisal=(L={incisal_l:.2f}, a={incisal_a:.2f}, b={incisal_b:.2f}), Middle=(L={middle_l:.2f}, a={middle_a:.2f}, b={middle_b:.2f}), Cervical=(L={cervical_l:.2f}, a={cervical_a:.2f}, b={cervical_b:.2f}), Overall=(L={overall_l:.2f}, a={overall_a:.2f}, b={overall_b:.2f})")

        # --- Call Enhanced Placeholder AI modules (now wrapped in try-except for robustness) ---
        try:
            # These functions will now also use random values for their simulated outputs
            face_features = detect_face_features(img_corrected) # Pass processed image for consistency, but logic is random
            tooth_analysis = segment_and_analyze_teeth(img_corrected) # Pass processed image for consistency, but logic is random
            aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis)
            print("DEBUG: Simulated AI modules executed.")
        except Exception as ai_module_error:
            print(f"WARNING: An error occurred during simulated AI module execution: {ai_module_error}")
            traceback.print_exc()


        # Create Colormath LAB objects from the generated random values
        incisal_lab_colormath = LabColor(lab_l=incisal_l, lab_a=incisal_a, lab_b=incisal_b)
        middle_lab_colormath = LabColor(lab_l=middle_l, lab_a=middle_a, lab_b=middle_b)
        cervical_lab_colormath = LabColor(lab_l=cervical_l, lab_a=cervical_a, lab_b=cervical_b)
        overall_lab_colormath = LabColor(lab_l=overall_l, lab_a=overall_a, lab_b=overall_b)
        
        print(f"DEBUG: Colormath LAB objects (from generated random values): Incisal={incisal_lab_colormath}, Middle={middle_lab_colormath}, Cervical={cervical_lab_colormath}, Overall={overall_lab_colormath}")

        # --- Delta E Matching (Step 6) ---
        incisal_delta_e_shade, incisal_min_delta = match_shade_with_delta_e(incisal_lab_colormath)
        middle_delta_e_shade, middle_min_delta = match_shade_with_delta_e(middle_lab_colormath)
        cervical_delta_e_shade, cervical_min_delta = match_shade_with_delta_e(cervical_lab_colormath)
        overall_delta_e_shade, overall_min_delta = match_shade_with_delta_e(overall_lab_colormath)
        
        print(f"DEBUG: Delta E matched shades: Overall={overall_delta_e_shade} (dE={overall_min_delta:.2f}), "
              f"Incisal={incisal_delta_e_shade} (dE={incisal_min_delta:.2f}), "
              f"Middle={middle_delta_e_shade} (dE={middle_min_delta:.2f}), "
              f"Cervical={cervical_delta_e_shade} (dE={cervical_min_delta:.2f})")

        # --- ML Prediction (using all 9 LAB features) ---
        overall_ml_shade = "Model Error" # Default to error
        if shade_classifier_model is not None:
            try:
                # Features for ML prediction: L, a, b for each of the three zones
                features_for_ml_prediction = np.array([[
                    incisal_lab_colormath.lab_l, incisal_lab_colormath.lab_a, incisal_lab_colormath.lab_b,
                    middle_lab_colormath.lab_l, middle_lab_colormath.lab_a, middle_lab_colormath.lab_b,
                    cervical_lab_colormath.lab_l, cervical_lab_colormath.lab_a, cervical_lab_colormath.lab_b
                ]])
                overall_ml_shade = shade_classifier_model.predict(features_for_ml_prediction)[0]
                print(f"DEBUG: Features for ML: {features_for_ml_prediction}")
                print(f"DEBUG: Predicted Overall Shade (ML): {overall_ml_shade}")
            except Exception as ml_predict_error:
                print(f"CRITICAL ERROR during ML prediction: {ml_predict_error}")
                traceback.print_exc()
                overall_ml_shade = "ML Prediction Failed" # More specific error state
        else:
            print("WARNING: AI model not loaded/trained. Cannot provide ML shade prediction.")

        # --- Calculate Confidence (Step 7) ---
        # Confidence calculation now uses a fixed "ideal" device profile assumption since reference correction is removed
        overall_accuracy_confidence, confidence_notes = calculate_confidence(overall_min_delta, "ideal")
        
        # Determine the final "Rule-based" shades using the Delta E matching (as per previous logic)
        final_incisal_rule_based = map_l_to_shade_rule_based(incisal_lab_colormath.lab_l, incisal_lab_colormath.lab_a, incisal_lab_colormath.lab_b)
        final_middle_rule_based = map_l_to_shade_rule_based(middle_lab_colormath.lab_l, middle_lab_colormath.lab_a, middle_lab_colormath.lab_b)
        final_cervical_rule_based = map_l_to_shade_rule_based(cervical_lab_colormath.lab_l, cervical_lab_colormath.lab_a, cervical_lab_colormath.lab_b)

        print(f"DEBUG: Final Rule-based Shades (to be displayed): Incisal={final_incisal_rule_based}, Middle={final_middle_rule_based}, Cervical={final_cervical_rule_based}")


        detected_shades = {
            "incisal": final_incisal_rule_based,
            "middle": final_middle_rule_based,
            "cervical": final_cervical_rule_based,
            "overall_ml_shade": overall_ml_shade,

            "delta_e_matched_shades": {
                "overall": overall_delta_e_shade,
                "overall_delta_e": round(float(overall_min_delta), 2),
                "incisal": incisal_delta_e_shade,
                "incisal_delta_e": round(float(incisal_min_delta), 2),
                "middle": middle_delta_e_shade,
                "middle_delta_e": round(float(middle_min_delta), 2),
                "cervical": cervical_delta_e_shade,
                "cervical_delta_e": round(float(cervical_min_delta), 2),
            },
            "face_features": face_features,
            "tooth_analysis": tooth_analysis,
            "aesthetic_suggestion": aesthetic_suggestion,
            "accuracy_confidence": {
                "overall_percentage": overall_accuracy_confidence,
                "notes": confidence_notes
            },
            "selected_device_profile": device_profile, # Still show what was selected
            "selected_reference_tab": reference_tab # Still show what was selected
        }
        return detected_shades

    except Exception as e:
        print(f"CRITICAL ERROR during shade detection: {e}")
        traceback.print_exc()
        return {
            "incisal": "Error", "middle": "Error", "cervical": "Error",
            "overall_ml_shade": "Error",
            "face_features": {},
            "tooth_analysis": {},
            "aesthetic_suggestion": {},
            "delta_e_matched_shades": {},
            "accuracy_confidence": {"overall_percentage": "N/A", "notes": f"Processing failed: {e}"},
            "selected_device_profile": device_profile, "selected_reference_tab": reference_tab
        }


def generate_pdf_report(patient_name, shades, image_path, filepath):
    """Generates a PDF report with detected shades and the uploaded image."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Shade View - Tooth Shade Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    selected_profile = shades.get("selected_device_profile", "N/A").replace("_", " ").title()
    pdf.cell(0, 10, txt=f"Simulated Device/Lighting Profile: {selected_profile}", ln=True)
    
    selected_ref_tab = shades.get("selected_reference_tab", "N/A").replace("_", " ").title()
    pdf.cell(0, 10, txt=f"Simulated Color Reference Used: {selected_ref_tab}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Detected Shades (Rule-based / Delta E):", ln=True) # Updated heading
    pdf.set_font("Arial", size=12)
    
    if "overall_ml_shade" in shades and shades["overall_ml_shade"] != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall AI Prediction (ML): {shades['overall_ml_shade']}", ln=True)

    pdf.cell(0, 7, txt=f"   - Incisal Zone (Rule-based): {shades['incisal']}", ln=True)
    pdf.cell(0, 7, txt=f"   - Middle Zone (Rule-based): {shades['middle']}", ln=True)
    pdf.cell(0, 7, txt=f"   - - Cervical Zone (Rule-based): {shades['cervical']}", ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Delta E 2000 Matched Shades (Perceptual Match):", ln=True)
    pdf.set_font("Arial", size=12)
    delta_e_shades = shades.get("delta_e_matched_shades", {})
    if delta_e_shades:
        pdf.cell(0, 7, txt=f"   - Overall Delta E Match: {delta_e_shades.get('overall', 'N/A')} (dE: {delta_e_shades.get('overall_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"   - Incisal Zone Delta E Match: {delta_e_shades.get('incisal', 'N/A')} (dE: {delta_e_shades.get('incisal_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"   - Middle Zone Delta E Match: {delta_e_shades.get('middle', 'N/A')} (dE: {delta_e_shades.get('middle_delta_e', 'N/A'):.2f})", ln=True)
        pdf.cell(0, 7, txt=f"   - Cervical Zone Delta E Match: {delta_e_shades.get('cervical', 'N/A')} (dE: {delta_e_shades.get('cervical_delta_e', 'N/A'):.2f})", ln=True)
    else:
        pdf.cell(0, 7, txt="   - Delta E matching data not available.", ln=True)


    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, txt="Shade Detection Accuracy Confidence:", ln=True)
    pdf.set_font("Arial", size=12)
    accuracy_conf = shades.get("accuracy_confidence", {})
    if accuracy_conf and accuracy_conf.get("overall_percentage") != "N/A":
        pdf.cell(0, 7, txt=f"   - Overall Confidence: {accuracy_conf.get('overall_percentage', 'N/A')}%", ln=True)
        pdf.multi_cell(0, 7, txt=f"   - Notes: {accuracy_conf.get('notes', 'N/A')}")
    else:
        pdf.cell(0, 7, txt="   - Confidence data not available or processing error.", ln=True)


    pdf.ln(8)
    pdf.set_font("Arial", 'B', size=13)
    pdf.cell(0, 10, txt="Advanced AI Insights (Simulated):", ln=True)
    pdf.set_font("Arial", size=11)

    tooth_analysis = shades.get("tooth_analysis", {})
    if tooth_analysis:
        pdf.cell(0, 7, txt="   -- Tooth Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Overall Shade (Detailed): {tooth_analysis.get('simulated_overall_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)
        l_val = tooth_analysis.get('overall_lab', {}).get('L', 'N/A')
        a_val = tooth_analysis.get('overall_lab', {}).get('a', 'N/A')
        b_val = tooth_analysis.get('overall_lab', {}).get('b', 'N/A')
        if all(isinstance(v, (int, float)) for v in [l_val, a_val, b_val]):
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val:.2f}, a={a_val:.2f}, b={b_val:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Overall LAB: L={l_val}, a={a_val}, b={b_val}", ln=True)
    
    pdf.ln(3)
    face_features = shades.get("face_features", {})
    if face_features:
        pdf.cell(0, 7, txt="   -- Facial Aesthetics Analysis --", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Skin Tone: {face_features.get('skin_tone', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Lip Color: {face_features.get('lip_color', 'N/A')}", ln=True)
        pdf.cell(0, 7, txt=f"   - Simulated Eye Contrast: {face_features.get('eye_contrast', 'N/A')}", ln=True)
        harmony_score = face_features.get('facial_harmony_score', 'N/A')
        if isinstance(harmony_score, (int, float)):
            pdf.cell(0, 7, txt=f"   - Simulated Facial Harmony Score: {harmony_score:.2f}", ln=True)
        else:
            pdf.cell(0, 7, txt=f"   - Simulated Facial Harmony Score: {harmony_score}", ln=True)

    pdf.ln(3)
    aesthetic_suggestion = shades.get("aesthetic_suggestion", {})
    if aesthetic_suggestion:
        pdf.cell(0, 7, txt=f"   -- Aesthetic Shade Suggestion --", ln=True)
        pdf.cell(0, 7, txt=f"   - Suggested Shade: {aesthetic_suggestion.get('suggested_aesthetic_shade', 'N/A')}",
                 ln=True)
        pdf.cell(0, 7, txt=f"   - Confidence: {aesthetic_suggestion.get('aesthetic_confidence', 'N/A')}", ln=True)
        pdf.multi_cell(0, 7, txt=f"   - Notes: {aesthetic_suggestion.get('recommendation_notes', 'N/A')}")

    pdf.ln(10)

    try:
        if os.path.exists(image_path):
            pdf.cell(0, 10, txt="Uploaded Image:", ln=True)
            if pdf.get_y() > 200:
                pdf.add_page()
            
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                h_img, w_img, _ = img_cv.shape
                max_w_pdf = 180
                w_pdf = min(w_img, max_w_pdf)
                h_pdf = h_img * (w_pdf / w_img)

                if pdf.get_y() + h_pdf + 10 > pdf.h - pdf.b_margin:
                    pdf.add_page()

                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                temp_image_path = "temp_pdf_image.png"
                cv2.imwrite(temp_image_path, img_rgb)
                
                pdf.image(temp_image_path, x=pdf.get_x(), y=pdf.get_y(), w=w_pdf, h=h_pdf)
                pdf.ln(h_pdf + 10)
                os.remove(temp_image_path)
            else:
                pdf.cell(0, 10, txt="Note: Image could not be loaded for embedding.", ln=True)

        else:
            pdf.cell(0, 10, txt="Note: Uploaded image file not found for embedding.", ln=True)
    except Exception as e:
        print(f"Error adding image to PDF: {e}")
        pdf.cell(0, 10, txt="Note: An error occurred while embedding the image in the report.", ln=True)

    pdf.set_font("Arial", 'I', size=9)
    pdf.multi_cell(0, 6,
                   txt="DISCLAIMER: This report is based on simulated analysis for demonstration purposes only. It is not intended for clinical diagnosis, medical advice, or professional cosmetic planning. Always consult with a qualified dental or medical professional for definitive assessment, diagnosis, and treatment.",
                   align='C')
    pdf.output(filepath)


# ===============================================
# 5. ROUTES (Adapted for Firestore)
# ===============================================
@app.route('/')
def home():
    """Renders the home/landing page."""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login (Simulated for Canvas)."""
    if g.user and 'anonymous' not in g.user['id']:
        flash(f"You are already logged in as {g.user['username']}.", 'info')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username or not password:
            error = 'Username and password are required.'

        if error is None:
            simulated_user_id = 'user_' + username.lower().replace(' ', '_')
            session['user_id'] = simulated_user_id
            session['user'] = {'id': simulated_user_id, 'username': username}
            flash(f'Simulated login successful for {username}!', 'success')
            print(f"DEBUG: Simulated login for user: {username} (ID: {session['user_id']})")
            return redirect(url_for('dashboard'))
        flash(error, 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration (Simulated for Canvas)."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            flash(f"Simulated registration successful for {username}. You can now log in!", 'success')
            return redirect(url_for('login'))
        flash(error, 'danger')

    return render_template('register.html')

@app.route('/logout')
def logout():
    """Handles user logout."""
    session.clear()
    flash('You have been logged out.', 'info')
    print(f"DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard with patient details and reports."""
    # Get patient details from SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE user_id = ?", (g.user['id'],))
    patients = cursor.fetchall()
    conn.close()

    # Format current date
    current_date_formatted = datetime.now().strftime('%Y-%m-%d')

    # Get all reports for the current Firestore user
    reports_collection_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
    all_reports = get_firestore_documents_in_collection(reports_collection_path)

    # Organize reports by OP number
    reports_by_op = {}
    for report in all_reports:
        op_number = report.get('op_number')
        if op_number not in reports_by_op:
            reports_by_op[op_number] = []
        reports_by_op[op_number].append(report)

    return render_template('dashboard.html',
                           patients=patients,
                           user=g.user,
                           current_date=current_date_formatted,
                           reports_by_op=reports_by_op)



@app.route('/save_patient_data', methods=['POST'])
@login_required
def save_patient_data():
    """Handles saving new patient records to SQLite and redirects to image upload page."""
    op_number = request.form['op_number']
    patient_name = request.form['patient_name']
    age = request.form['age']
    sex = request.form['sex']
    record_date = request.form['date']
    user_id = g.user['id']

    conn = get_db_connection()

    # Check if OP Number already exists for this user
    existing_patient = conn.execute(
        'SELECT * FROM patients WHERE op_number = ? AND user_id = ?',
        (op_number, user_id)
    ).fetchone()

    if existing_patient:
        flash('OP Number already exists for another patient under your account. Please use a unique OP Number.', 'danger')
        conn.close()
        return redirect(url_for('dashboard'))

    try:
        conn.execute(
            '''
            INSERT INTO patients (user_id, op_number, patient_name, age, sex, record_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (user_id, op_number, patient_name, age, sex, record_date, datetime.now().isoformat())
        )
        conn.commit()
        flash('Patient record saved successfully to the database! Now upload an image.', 'success')
        return redirect(url_for('upload_page', op_number=op_number))
    except Exception as e:
        flash(f'Error saving patient record to database: {e}', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        conn.close()



@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    """Renders the dedicated image upload page for a specific patient."""
    user_id = g.user['id']

    patients_collection_path = ['artifacts', app_id, 'users', user_id, 'patients']
    patient = None
    all_patients = get_firestore_documents_in_collection(patients_collection_path, query_filters={'op_number': op_number, 'user_id': user_id})
    if all_patients:
        patient = all_patients[0]

    if patient is None:
        flash('Patient not found or unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    return render_template('upload_page.html', op_number=op_number, patient_name=patient['patient_name'])


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handles image upload, shade detection, and PDF report generation."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        op_number_from_form = request.form.get('op_number')
        patient_name = request.form.get('patient_name', 'Unnamed Patient')
        device_profile = request.form.get('device_profile', 'ideal')
        reference_tab = request.form.get('reference_tab', 'neutral_gray')

        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1]
            unique_filename = str(uuid.uuid4()) + file_ext
            
            original_image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(original_image_path)
            flash('Image uploaded successfully!', 'success')

            detected_shades = detect_shades_from_image(original_image_path, device_profile, reference_tab)

            if (detected_shades.get("incisal") == "N/A" and
                detected_shades.get("middle") == "N/A" and
                detected_shades.get("cervical") == "N/A" and
                detected_shades.get("overall_ml_shade") == "N/A" and
                detected_shades.get("delta_e_matched_shades", {}).get("overall") == "N/A"):
                flash("Error processing image for shade detection. Please try another image or check image quality.", 'danger')
                if os.path.exists(original_image_path):
                    os.remove(original_image_path)
                return redirect(url_for('upload_page', op_number=op_number_from_form))


            report_filename = f"report_{patient_name.replace(' ', '')}{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_filepath = os.path.join(REPORT_FOLDER, report_filename)
            generate_pdf_report(patient_name, detected_shades, original_image_path, report_filepath)
            flash('PDF report generated!', 'success')

            formatted_analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            report_data = {
                'patient_name': patient_name,
                'op_number': op_number_from_form,
                'original_image': unique_filename,
                'report_filename': report_filename,
                'detected_shades': detected_shades,
                'timestamp': datetime.now().isoformat(),
                'user_id': g.firestore_user_id
            }
            reports_collection_path = ['artifacts', app_id, 'users', g.firestore_user_id, 'reports']
            add_firestore_document(reports_collection_path, report_data)

            return render_template('report.html',
                                   patient_name=patient_name,
                                   shades=detected_shades,
                                   image_filename=unique_filename,
                                   report_filename=report_filename,
                                   analysis_date=formatted_analysis_date,
                                   device_profile=device_profile,
                                   reference_tab=reference_tab)
    
    flash("Please select a patient from the dashboard to upload an image.", 'info')
    return redirect(url_for('dashboard'))


@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Allows downloading of generated PDF reports."""
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/ux_report')
def ux_report_page():
    """Renders the UX Report page."""
    return render_template('ux_report.html')

if __name__ == '__main__':
    if shade_classifier_model is None:
        print("CRITICAL: Machine Learning model could not be loaded or trained. Shade prediction will not work.")
    init_db()
    app.run(debug=True)
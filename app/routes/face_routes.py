# File: app/routes/face_routes.py

from datetime import datetime, timedelta
import json
import time
import cv2
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, Response, stream_with_context
from flask_login import login_required, current_user
import numpy as np
from sqlalchemy.sql import text
import os
import joblib
import queue
import csv
# Import Model Database Anda
from app.models import Personnels, Personnel_Images, Camera_Settings, Work_Timer, Personnel_Entries, Company, Divisions # Ensure Work_Timer has TYPE_FACE_DETECTED
from app import db # Import instance db Anda

# Import untuk decorators
from app.utils.decorators import employee_required, admin_required # Ensure these exist

# Import untuk model AI baru
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN # <<< MTCNN IMPORTED
from torchvision import transforms
from PIL import Image # Pillow untuk konversi gambar

# Import untuk perbandingan embedding
from sklearn.metrics.pairwise import cosine_similarity
# Flask, Response, cv2, time already imported via Flask and cv2

# (app = Flask(__name__) is not needed here as 'bp' will be registered with the main Flask app)

# Cache untuk instance kamera agar tidak membuka berkali-kali
_camera_instance_cache = {}

bp = Blueprint('face', __name__, template_folder='../templates')

# ====================================================================
# Global AI/CV Settings & Initialization
# ====================================================================
_FALLBACK_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DEFAULT_TRAINED_MODELS_PATH_FALLBACK = os.path.join(_FALLBACK_APP_ROOT, 'app', 'trained_models')
_DEFAULT_UPLOAD_FOLDER_FALLBACK = os.path.join(_FALLBACK_APP_ROOT, 'app', 'static', 'img') # For personnel_pics

# --- MTCNN and Embeddings Settings ---
EMBEDDINGS_DB_FILENAME = 'face_embeddings_mtcnn_resnet_torch.joblib' # Reflects MTCNN use
face_detector = None  # Will be MTCNN instance
resnet_embedder = None
device = None
known_embeddings_db = {"embeddings": np.array([]), "ids": []}
EMBEDDINGS_DB_PATH = "" # Akan diisi di initialize_ai_models

# --- Presence Queue ---
presence_message_queue = queue.Queue(maxsize=100) # Max 100 messages

# --- Default Constants (can be overridden by app.config) ---
# MTCNN Specific
MIN_MTCNN_CONFIDENCE_DATASET = 0.95 # For capturing dataset images (higher quality)
MIN_MTCNN_CONFIDENCE_STREAM = 0.90  # For live stream detection/recognition
MTCNN_FACE_MARGIN = 20              # Margin around detected face for ROI cropping (pixels)
MTCNN_MIN_FACE_SIZE = 30            # Minimum face size (pixels) MTCNN will try to detect
MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]  # PNet, RNet, ONet thresholds for MTCNN
MTCNN_FACTOR = 0.709                # MTCNN scale factor

# Recognition & General
COSINE_SIMILARITY_THRESHOLD = 0.7
RECOGNITION_FRAME_INTERVAL = 1     # Process every Nth frame for full AI pipeline in streams
MAX_DATASET_IMAGES_PER_CAPTURE = 50 # Max images to capture in one go for dataset
WORK_TIMER_SAVE_INTERVAL_MINUTES = 0.1
RECOGNITION_AND_PRESENCE_INTERVAL_SECONDS = 1 # Interval for presence detection attempts in presence stream
MIN_SECONDS_BETWEEN_PRESENCE_ATTEMPTS_PERSONNEL = 20 # Cooldown for same person presence

preprocess_transform = None # For ResNet embedder

# Recognition & General

MIN_MTCNN_CONFIDENCE_STREAM = 0.9 # Set a default for MTCNN confidence

detection_times = {} # For Work Timer: {personnel_name: accumulated_seconds}
last_detection_time = {} # For Work Timer: {personnel_name: last_seen_datetime}
last_save_time_global_work_timer = datetime.min # For Work Timer DB save interval
last_save_time_per_person = {}

# --- Shared Global State for Presence (Presence Stream Specific) ---
last_presence_attempt_time_personnel = {} # {personnel_id: last_attempt_datetime}


CACHE_FILE = "embedding_cache.json"

def load_embedding_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_embedding_cache(cache_data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)

def get_personnel_folder_path_local():
    if current_app:
        upload_folder = current_app.config.get('UPLOAD_FOLDER', _DEFAULT_UPLOAD_FOLDER_FALLBACK)
    else:
        upload_folder = _DEFAULT_UPLOAD_FOLDER_FALLBACK
    return os.path.join(upload_folder, 'personnel_pics')

def get_upload_folder_presence_proofs(): # Specific for presence proofs
    if current_app:
        base_upload_folder = current_app.config.get('UPLOAD_FOLDER', _DEFAULT_UPLOAD_FOLDER_FALLBACK)
        # Assuming UPLOAD_FOLDER is typically <app_root>/app/static/img
        # Presence proofs might go into a subfolder of UPLOAD_FOLDER or a separate one defined in config
        return current_app.config.get('UPLOAD_FOLDER_PRESENCE_PROOFS', os.path.join(base_upload_folder, 'presence_proofs'))
    else:
        return os.path.join(_DEFAULT_UPLOAD_FOLDER_FALLBACK, 'presence_proofs')


def get_config_val(key, default_value, data_type=None):
    val = current_app.config.get(key, default_value) if current_app else default_value
    if data_type:
        try:
            return data_type(val)
        except (ValueError, TypeError):
            current_app.logger.warning(f"Config Error: Could not cast {key}='{val}' to {data_type}. Using default: {default_value}")
            return default_value
    return val

def clean_name_for_filename(name_str):
    name_str = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '' for c in str(name_str))
    return name_str.replace(' ', '_')

# Menginisialisasi model AI (MTCNN + ResNet) dan mengatur konfigurasi awal
def initialize_ai_models(app_reference): 
    global face_detector, resnet_embedder, device, known_embeddings_db, preprocess_transform
    global EMBEDDINGS_DB_PATH
    global MIN_MTCNN_CONFIDENCE_DATASET, MIN_MTCNN_CONFIDENCE_STREAM, MTCNN_FACE_MARGIN, MTCNN_MIN_FACE_SIZE, MTCNN_THRESHOLDS, MTCNN_FACTOR
    global COSINE_SIMILARITY_THRESHOLD, RECOGNITION_FRAME_INTERVAL, MAX_DATASET_IMAGES_PER_CAPTURE
    global WORK_TIMER_SAVE_INTERVAL_MINUTES, RECOGNITION_AND_PRESENCE_INTERVAL_SECONDS, MIN_SECONDS_BETWEEN_PRESENCE_ATTEMPTS_PERSONNEL

    logger = app_reference.logger

    logger.info("Attempting to initialize AI models (MTCNN + ResNet)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"AI models will use device: {device}")

    trained_models_path = get_config_val('TRAINED_MODELS_PATH', _DEFAULT_TRAINED_MODELS_PATH_FALLBACK)
    EMBEDDINGS_DB_PATH = os.path.join(trained_models_path, EMBEDDINGS_DB_FILENAME)
    os.makedirs(trained_models_path, exist_ok=True)
    os.makedirs(get_upload_folder_presence_proofs(), exist_ok=True) # Ensure presence proof folder exists

    # Load MTCNN specific configurations
    MIN_MTCNN_CONFIDENCE_DATASET = get_config_val('MIN_MTCNN_CONFIDENCE_DATASET', MIN_MTCNN_CONFIDENCE_DATASET, float)
    MIN_MTCNN_CONFIDENCE_STREAM = get_config_val('MIN_MTCNN_CONFIDENCE_STREAM', MIN_MTCNN_CONFIDENCE_STREAM, float)
    MTCNN_FACE_MARGIN = get_config_val('MTCNN_FACE_MARGIN', MTCNN_FACE_MARGIN, int)
    MTCNN_MIN_FACE_SIZE = get_config_val('MTCNN_MIN_FACE_SIZE', MTCNN_MIN_FACE_SIZE, int)

    # Load general configurations
    COSINE_SIMILARITY_THRESHOLD = get_config_val('COSINE_SIMILARITY_THRESHOLD', COSINE_SIMILARITY_THRESHOLD, float)
    RECOGNITION_FRAME_INTERVAL = get_config_val('RECOGNITION_FRAME_INTERVAL', RECOGNITION_FRAME_INTERVAL, int)
    MAX_DATASET_IMAGES_PER_CAPTURE = get_config_val('MAX_DATASET_IMAGES_PER_CAPTURE', MAX_DATASET_IMAGES_PER_CAPTURE, int)
    WORK_TIMER_SAVE_INTERVAL_MINUTES = get_config_val('WORK_TIMER_SAVE_INTERVAL_MINUTES', WORK_TIMER_SAVE_INTERVAL_MINUTES, int)
    RECOGNITION_AND_PRESENCE_INTERVAL_SECONDS = get_config_val('RECOGNITION_AND_PRESENCE_INTERVAL_SECONDS', RECOGNITION_AND_PRESENCE_INTERVAL_SECONDS, int)
    MIN_SECONDS_BETWEEN_PRESENCE_ATTEMPTS_PERSONNEL = get_config_val('MIN_SECONDS_BETWEEN_PRESENCE_ATTEMPTS_PERSONNEL', MIN_SECONDS_BETWEEN_PRESENCE_ATTEMPTS_PERSONNEL, int)

    preprocess_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        if face_detector is None:
            face_detector = MTCNN(
                keep_all=True,      
                device=device,
                margin=MTCNN_FACE_MARGIN,
                min_face_size=MTCNN_MIN_FACE_SIZE,
                thresholds=MTCNN_THRESHOLDS,
                factor=MTCNN_FACTOR,
                post_process=True,   
                select_largest=False  
            )
            logger.info(f"MTCNN Face Detector initialized on {device}.")
        
        if resnet_embedder is None:
            resnet_embedder = InceptionResnetV1(pretrained='vggface2', device=device).eval()
            logger.info(f"InceptionResnetV1 (VGGFace2) embedder loaded on {device}.")
        
        load_known_embeddings()
    except Exception as e:
        print(f"FATAL ERROR during AI model initialization: {e}", exc_info=True)
        if current_app: flash(f"Failed to initialize AI models: {e}. Contact admin.", "error")

# ====================================================================
# Helper Functions AI (MTCNN + PyTorch InceptionResnetV1)
# ====================================================================
# Mengekstrak ROI wajah dari frame BGR menggunakan koordinat kotak MTCNN [x1, y1, x2, y2]
def get_face_roi_mtcnn(frame_bgr, box_coords_input): 

    logger = current_app.logger 

    if frame_bgr is None or box_coords_input is None:
        logger.debug("get_face_roi_mtcnn: frame_bgr or box_coords_input is None.")
        return None

    if not isinstance(box_coords_input, np.ndarray):
        try:
         
            box_coords_numeric = np.array(box_coords_input, dtype=np.float32)
            logger.debug(f"get_face_roi_mtcnn: box_coords_input converted from list/tuple to ndarray: {box_coords_numeric}")
        except Exception as e_conv_array:
            logger.warning(f"get_face_roi_mtcnn: Failed to convert box_coords_input to ndarray: {e_conv_array}. Input: {box_coords_input}")
            return None
    elif box_coords_input.dtype == object or box_coords_input.dtype.kind not in 'fiu': 
        try:
            box_coords_numeric = box_coords_input.astype(np.float32)
            logger.debug(f"get_face_roi_mtcnn: box_coords_input (dtype {box_coords_input.dtype}) cast to float32: {box_coords_numeric}")
        except (ValueError, TypeError) as e_cast:
            logger.warning(f"get_face_roi_mtcnn: Failed to cast box_coords_input (dtype {box_coords_input.dtype}) to float32: {e_cast}. Input: {box_coords_input}")
            return None
    else:
        box_coords_numeric = box_coords_input.astype(np.float32, copy=False)

    try:
        if np.any(np.isnan(box_coords_numeric)) or np.any(np.isinf(box_coords_numeric)):
            logger.debug(f"get_face_roi_mtcnn: Invalid (NaN/Inf) values in numeric box coordinates: {box_coords_numeric}")
            return None

        if box_coords_numeric.shape != (4,):
            logger.warning(f"get_face_roi_mtcnn: box_coords_numeric does not have 4 elements: {box_coords_numeric}")
            return None

        x1, y1, x2, y2 = [int(c) for c in box_coords_numeric]
        
        if x1 >= x2 or y1 >= y2: 
            logger.debug(f"get_face_roi_mtcnn: Invalid box dimensions after int conversion: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None

        frame_h, frame_w = frame_bgr.shape[:2]

        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(frame_w, x2)
        y2_clamped = min(frame_h, y2)

        if x1_clamped >= x2_clamped or y1_clamped >= y2_clamped:
            logger.debug(f"get_face_roi_mtcnn: Box after clamping is invalid: {x1_clamped, y1_clamped, x2_clamped, y2_clamped}")
            return None
            
        face_roi = frame_bgr[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        
        if face_roi.size == 0:
            logger.debug(f"get_face_roi_mtcnn: Extracted ROI is empty. Original numeric box: {box_coords_numeric}, Clamped: {(x1_clamped, y1_clamped, x2_clamped, y2_clamped)}")
            return None
            
        return face_roi
    except Exception as e: 
        print(f"get_face_roi_mtcnn: Unexpected error processing box: {e}. Numeric Box: {box_coords_numeric if 'box_coords_numeric' in locals() else 'N/A'}", exc_info=True)
        return None
    
# Melakukan preprocessing pada ROI wajah BGR (NumPy array) untuk ResNet embedder
def preprocess_face_for_resnet_pytorch(face_roi_bgr):
    global preprocess_transform 
    logger = current_app.logger if current_app else print

    if face_roi_bgr is None or face_roi_bgr.size == 0:
        logger.debug("preprocess_face: input face_roi_bgr is None or empty.")
        return None
    if preprocess_transform is None:
        print("preprocess_face: preprocess_transform is not initialized!")
        return None
    try:
        face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_roi_rgb)
        return preprocess_transform(pil_image)
    except Exception as e:
        print(f"Error in preprocess_face_for_resnet_pytorch: {e}", exc_info=True)
        return None

#  Mengambil embedding wajah menggunakan ResNet InceptionV1
def get_embedding_resnet_pytorch(face_tensor_transformed):
    global resnet_embedder, device
    if resnet_embedder is None or face_tensor_transformed is None:
        return None
    try:
       
        if face_tensor_transformed.ndim == 3:
            face_tensor_transformed = face_tensor_transformed.unsqueeze(0)
        
        face_tensor_transformed = face_tensor_transformed.to(device)
        with torch.no_grad():
            embedding = resnet_embedder(face_tensor_transformed)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        # current_app.print(f"Error in get_embedding_resnet_pytorch: {e}", exc_info=True)
        return None

#  Memuat embeddings yang sudah ada dari file
def load_known_embeddings():
    global known_embeddings_db, EMBEDDINGS_DB_PATH
    logger = current_app.logger if current_app else print
    if not EMBEDDINGS_DB_PATH:
        print("EMBEDDINGS_DB_PATH not set. Cannot load embeddings.")
        known_embeddings_db = {"embeddings": np.array([]), "ids": []}
        return
    try:
        if os.path.exists(EMBEDDINGS_DB_PATH):
            data = joblib.load(EMBEDDINGS_DB_PATH)
            if isinstance(data, dict) and "embeddings" in data and "ids" in data and \
               isinstance(data["embeddings"], np.ndarray) and isinstance(data["ids"], list):
                known_embeddings_db = data
                logger.info(f"Known embeddings loaded: {len(known_embeddings_db['ids'])} personnel, {known_embeddings_db['embeddings'].shape[0]} embeddings from {EMBEDDINGS_DB_PATH}.")
            else:
                known_embeddings_db = {"embeddings": np.array([]), "ids": []}
                logger.warning(f"Embeddings DB at {EMBEDDINGS_DB_PATH} format error. Reinitialized.")
        else:
            known_embeddings_db = {"embeddings": np.array([]), "ids": []}
            logger.info(f"No embeddings DB at {EMBEDDINGS_DB_PATH}. Initialized new.")
    except Exception as e:
        print(f"Error loading embeddings DB from {EMBEDDINGS_DB_PATH}: {e}", exc_info=True)
        known_embeddings_db = {"embeddings": np.array([]), "ids": []}
        
# Menyimpan embeddings yang sudah ada ke file
def save_known_embeddings():
    global known_embeddings_db, EMBEDDINGS_DB_PATH
    logger = current_app.logger if current_app else print
    if not EMBEDDINGS_DB_PATH:
        print("EMBEDDINGS_DB_PATH not set. Cannot save embeddings.")
        return
    try:
        os.makedirs(os.path.dirname(EMBEDDINGS_DB_PATH), exist_ok=True)
        joblib.dump(known_embeddings_db, EMBEDDINGS_DB_PATH)
        logger.info(f"Known embeddings saved to {EMBEDDINGS_DB_PATH}")
    except Exception as e:
        print(f"Error saving embeddings DB: {e}", exc_info=True)

# ====================================================================
# Logika Pengambilan Gambar Wajah (Dataset Capture) - MTCNN
# ====================================================================
def capture_faces_logic(personnel):
    global face_detector, MIN_MTCNN_CONFIDENCE_DATASET, MAX_DATASET_IMAGES_PER_CAPTURE
    logger = current_app.logger

    if face_detector is None:
        print("MTCNN Face Detector not initialized for capture_faces_logic.")
        return {'status': 'error', 'message': 'MTCNN Face Detector not initialized.'}

    personnel_id_str = str(personnel.id)
    personnel_name_folder_safe = personnel.name
    target_folder_for_personnel = os.path.join(get_personnel_folder_path_local(), personnel_name_folder_safe)
    os.makedirs(target_folder_for_personnel, exist_ok=True)

    cap = get_camera_instance(0) 
    if not cap or not cap.isOpened():
        print("Failed to open camera for dataset capture.")
        return {'status': 'error', 'message': 'Failed to open camera for capture.'}

    existing_files_count = len([name for name in os.listdir(target_folder_for_personnel) if os.path.isfile(os.path.join(target_folder_for_personnel, name))])
    captured_this_session = 0
    base_db_path_prefix = os.path.join('personnel_pics', personnel_name_folder_safe)
    capture_attempts = 0
    MAX_CAPTURE_ATTEMPTS = 20

    logger.info(f"Starting face capture for {personnel.name}. Target: {MAX_DATASET_IMAGES_PER_CAPTURE} images.")

    try:
        while captured_this_session < MAX_DATASET_IMAGES_PER_CAPTURE and capture_attempts < MAX_CAPTURE_ATTEMPTS:
            capture_attempts += 1
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                time.sleep(0.05)
                continue

            # Convert BGR to RGB PIL Image for MTCNN
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
            except Exception as e_conv:
                logger.warning(f"Dataset Capture: Error converting frame to PIL: {e_conv}")
                continue
            
            # Detect faces using MTCNN
            boxes, probs = face_detector.detect(frame_pil)

            if boxes is not None and len(boxes) > 0:

                best_idx = np.argmax(probs)
                best_box = boxes[best_idx]
                best_prob = probs[best_idx]

                if best_prob >= MIN_MTCNN_CONFIDENCE_DATASET:
                    face_roi_bgr_to_save = get_face_roi_mtcnn(frame_bgr, best_box)
                    
                    if face_roi_bgr_to_save is not None and face_roi_bgr_to_save.size > 0:
                        existing_files_count += 1
                        file_name = f"face_{personnel_id_str}_{personnel_name_folder_safe}_{existing_files_count:03d}.jpg"
                        file_path_abs = os.path.join(target_folder_for_personnel, file_name)
                        
                        try:
                            cv2.imwrite(file_path_abs, face_roi_bgr_to_save)
                            captured_this_session += 1
                            
                            db_image_path = os.path.join(base_db_path_prefix, file_name).replace("\\", "/")
                            new_personnel_image = Personnel_Images(personnel_id=personnel.id, image_path=db_image_path)
                            db.session.add(new_personnel_image)
                            logger.info(f"Captured face {captured_this_session}/{MAX_DATASET_IMAGES_PER_CAPTURE} for {personnel.name}: {file_name} (Prob: {best_prob:.2f})")
                            time.sleep(0.5) # Pause slightly to allow for different poses
                        except Exception as e_write:
                            print(f"Error writing image {file_name}: {e_write}")
                            existing_files_count -=1 
            time.sleep(0.05) 
    finally:
        release_camera_instance(0) 

    try:
        if captured_this_session > 0:
            db.session.commit()
            # _train_face_model_logic()
            return {'status': 'success', 'message': f'Face capture completed for {personnel.name}. Captured {captured_this_session} new faces.'}
        else:
            db.session.rollback()
            return {'status': 'warning', 'message': 'No new faces captured. Ensure face is clearly visible and well-lit.'}
    except Exception as e_commit:
        db.session.rollback()
        print(f"DB commit error after face capture: {e_commit}", exc_info=True)
        return {'status': 'error', 'message': 'Database error saving captured faces.'}


# ====================================================================
# Logika "Training" (Pembuatan Database Embedding) - MTCNN + RESNET
# ====================================================================
def _train_face_model_logic():
    global known_embeddings_db, face_detector, resnet_embedder, MIN_MTCNN_CONFIDENCE_DATASET
    logger = current_app.logger

    if face_detector is None or resnet_embedder is None:
        print("AI models not initialized for training.")
        return {'status': 'error', 'message': 'AI models not initialized.', 'success': False}

    personnel_base_folder = get_personnel_folder_path_local()
    if not os.path.exists(personnel_base_folder):
        print(f"Dataset folder missing: {personnel_base_folder}")
        return {'status': 'error', 'message': f'Dataset folder missing: {personnel_base_folder}', 'success': False}

    # Load old cache
    cache_data = load_embedding_cache()

    # Load previous known embeddings
    embeddings_list = []
    ids_list = []

    if known_embeddings_db and 'embeddings' in known_embeddings_db and 'ids' in known_embeddings_db:
        embeddings_list = list(known_embeddings_db['embeddings'])
        ids_list = list(known_embeddings_db['ids'])

    # DB personnel
    personnel_name_to_id_map = {p.name: str(p.id) for p in Personnels.query.all()}
    logger.info(f"Starting incremental embedding generation. Found {len(personnel_name_to_id_map)} personnel in DB.")

    all_personnel_dirs = [d for d in os.listdir(personnel_base_folder) if os.path.isdir(os.path.join(personnel_base_folder, d))]

    updated_cache_data = dict(cache_data)  # shallow copy
    new_images_processed = 0

    for person_name_folder in all_personnel_dirs:
        person_path = os.path.join(personnel_base_folder, person_name_folder)
        person_id_from_folder = personnel_name_to_id_map.get(person_name_folder)

        if not person_id_from_folder:
            logger.warning(f"Skipping folder '{person_name_folder}', no matching active personnel found in DB by name.")
            continue

        # Existing cache for this person
        person_cache = cache_data.get(person_name_folder, {})

        updated_person_cache = dict(person_cache)
        new_images_for_person = 0

        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            last_modified = os.path.getmtime(img_path)

            # Check if new or changed
            if img_file in person_cache and abs(person_cache[img_file] - last_modified) < 1:
                # Skip already processed
                continue

            try:
                frame_bgr = cv2.imread(img_path)
                if frame_bgr is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue

                frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                boxes, probs = face_detector.detect(frame_pil)

                if boxes is not None and len(boxes) > 0:
                    best_idx = np.argmax(probs)
                    box = boxes[best_idx]

                    if probs[best_idx] >= MIN_MTCNN_CONFIDENCE_DATASET:
                        face_roi = get_face_roi_mtcnn(frame_bgr, box)
                        if face_roi is None or face_roi.size == 0: continue

                        face_tensor = preprocess_face_for_resnet_pytorch(face_roi)
                        if face_tensor is None: continue

                        embedding = get_embedding_resnet_pytorch(face_tensor)
                        if embedding is not None:
                            embeddings_list.append(embedding)
                            ids_list.append(person_id_from_folder)
                            new_images_for_person += 1
                            updated_person_cache[img_file] = last_modified
                        else:
                            logger.warning(f"Embedding generation failed for {img_file}")
                else:
                    logger.warning(f"No face detected in {img_file}")
            except Exception as e:
                logger.error(f"Error processing {img_file} in {person_name_folder}: {e}", exc_info=True)

        if new_images_for_person > 0:
            new_images_processed += new_images_for_person
            logger.info(f"Added {new_images_for_person} new embeddings for {person_name_folder}")
        
        updated_cache_data[person_name_folder] = updated_person_cache

    if new_images_processed > 0:
        # Save updated embeddings
        known_embeddings_db = {"embeddings": np.array(embeddings_list), "ids": ids_list}
        save_known_embeddings()
        save_embedding_cache(updated_cache_data)
        logger.info(f"Incremental training complete. Added {new_images_processed} new embeddings.")
        return {'status': 'success', 'message': f'Added {new_images_processed} new embeddings.', 'success': True}
    else:
        logger.info("No new images to process.")
        return {'status': 'success', 'message': 'No new images to process.', 'success': True}


# ====================================================================
# Streaming for WORK TIMER (Recognition and Time Tracking)
# ====================================================================
total_frames_processed = 0
total_faces_detected = 0
total_faces_recognized = 0
total_faces_unknown = 0
last_report_time = time.time() # Untuk interval pelaporan/penyimpanan

## --- CSV LOGGING CONFIGURATION ---
CSV_LOG_DIR = 'accuracy_logs'
os.makedirs(CSV_LOG_DIR, exist_ok=True)
LOG_INTERVAL_SECONDS = 15

# --- GLOBAL METRICS (for accumulation within a logging interval) ---
last_log_time = time.time()
last_frame_time = time.time()
detection_times = {}
last_detection_time = {}
last_save_time_per_person = {}

current_interval_total_frames = 0
current_interval_frames_with_faces = 0 
current_interval_total_faces_detected = 0
current_interval_total_faces_recognized = 0
current_interval_total_faces_unknown = 0
current_interval_recognition_scores = []
current_interval_fps_values = []

# --- MODIFIED log_metrics_to_csv FUNCTION ---
def log_metrics_to_csv(
    camera_url: str,
    scenario_type: str,
    condition_variation: str,
    jml_percobaan: int,
    true_positives: int,
    false_negatives: int,
    detection_rate: float,
    false_negative_rate: float,
    notes: str
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(CSV_LOG_DIR, "work_timer_face_metrics.csv")

    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp", "Camera URL", "Skenario Pengujian", "Kondisi/Variasi",
                "Jml Percobaan", "True Positives", "False Negatives",
                "Detection Rate (TPR)", "False Negative Rate", "Catatan"
            ])

        writer.writerow([
            timestamp,
            camera_url,
            scenario_type,
            condition_variation,
            jml_percobaan,
            true_positives,
            false_negatives,
            f"{detection_rate:.2f}%",
            f"{false_negative_rate:.2f}%",
            notes
        ])
    print(f"Performance metrics saved to {file_path}")

def is_personnel_from_logged_in_company(personnel_company_id, current_user_id, db_session):
    if not personnel_company_id or not current_user_id:
        return False

    try:
        company = Company.query.filter_by(user_id=current_user_id).first()

        return company.id == personnel_company_id
    except Exception as e:
        print(f"[Validation] Error checking match: {e}")

    return False

def log_face_detection_metrics_to_csv(
    name: str,
    scenario_type: str,
    avg_fps: float,
    jml_percobaan: int,
    true_positives: int,
    false_negatives: int,
    detection_rate: float,
    false_negative_rate: float,
    notes: str
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(CSV_LOG_DIR, "face_detection_metrics.csv")
    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp", "Nama", "Skenario Pengujian", "FPS",
                "Jml Percobaan", "True Positives", "False Negatives",
                "Detection Rate (TPR)", "False Negative Rate", "Catatan"
            ])

        writer.writerow([
            timestamp,
            name,
            scenario_type,
            f"{avg_fps:.2f}",
            jml_percobaan,
            true_positives,
            false_negatives,
            f"{detection_rate:.2f}%",
            f"{false_negative_rate:.2f}%",
            notes
        ])

def log_face_recognition_metrics_to_csv(
    name: str,
    scenario_type: str,
    jml_percobaan: int,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    detection_rate: float,
    false_negative_rate: float,
    avg_confidence: float,
    notes: str
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(CSV_LOG_DIR, "face_recognition_metrics.csv")
    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp", "Nama", "Skenario Pengujian",
                "Jml Percobaan", "True Positives (Dikenali Benar)",
                "False Positives (Dikenali Salah)", "False Negatives (Unknown)",
                "Recognition Rate (TPR)", "False Negative Rate",
                "Avg Confidence", "Catatan"
            ])

        writer.writerow([
            timestamp,
            name,
            scenario_type,
            jml_percobaan,
            true_positives,
            false_positives,
            false_negatives,
            f"{detection_rate:.2f}%",
            f"{false_negative_rate:.2f}%",
            f"{avg_confidence:.2f}",
            notes
        ])


# --- MODIFIED generate_work_timer_stream_frames FUNCTION ---
def generate_work_timer_stream_frames(camera_source_url: str, cam_settings_obj: Camera_Settings, app_context_obj, db_session_obj):
    global last_log_time, detection_times, last_detection_time, last_save_time_per_person
    global current_interval_total_frames, current_interval_frames_with_faces, \
           current_interval_total_faces_detected, current_interval_total_faces_recognized, \
           current_interval_total_faces_unknown, current_interval_recognition_scores, \
           current_interval_fps_values, last_frame_time, current_interval_total_faces_false_positive, scenarios

    # Reset metrics for the *start* of this specific stream
    current_interval_total_frames = 0
    current_interval_frames_with_faces = 0
    current_interval_total_faces_detected = 0
    current_interval_total_faces_recognized = 0
    current_interval_total_faces_false_positive = 0
    current_interval_total_faces_unknown = 0
    current_interval_recognition_scores = []
    current_interval_fps_values = []
    scenarios = []
    
    last_log_time = time.time()
    last_frame_time = time.time() 
    logged_in_user_id = None

    company_obj = Company.query.filter_by(user_id=current_user.id).first() 
    user_company_id = company_obj.id

    print(f"[WorkTimerStream {camera_source_url}] Starting stream processing.")
    

    if face_detector is None or resnet_embedder is None:
        error_msg = "AI MODEL NOT READY. Please check initialization."
        print(f"[WorkTimerStream {camera_source_url}] {error_msg}")
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, error_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    if known_embeddings_db["embeddings"].size == 0:
        error_msg = "EMBEDDINGS DATABASE EMPTY. No known faces to recognize."
        print(f"[WorkTimerStream {camera_source_url}] {error_msg}")
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, error_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    cap = get_camera_instance(camera_source_url)
    if not cap or not cap.isOpened():
        error_msg = f"CAMERA ERROR: Could not open {camera_source_url}"
        print(f"[WorkTimerStream {camera_source_url}] {error_msg}")
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(frame, error_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        return

    personnel_map = {}
    try:
        all_personnels = db_session_obj.query(Personnels).all()
        
        # Simpan: ID -> {name, company_id}
        personnel_map = {
            str(p.id): {
                "name": p.name,
                "company_id": p.company_id
            }
            for p in all_personnels
        }

        print(f"[WorkTimerStream {camera_source_url}] Loaded {len(personnel_map)} personnels from DB.")
    except Exception as e_map:
        print(f"[WorkTimerStream {camera_source_url}] Error loading personnel map from DB: {e_map}")
        personnel_map = {}

    frame_idx = 0
    consecutive_read_errors = 0
    MAX_CONSECUTIVE_READ_ERRORS_STREAM = 60

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                consecutive_read_errors += 1
                print(f"[WorkTimerStream {camera_source_url}] Frame read failed ({consecutive_read_errors}/{MAX_CONSECUTIVE_READ_ERRORS_STREAM}).")
                if consecutive_read_errors > MAX_CONSECUTIVE_READ_ERRORS_STREAM:
                    print(f"[WorkTimerStream {camera_source_url}] Max read errors reached. Stopping stream.")
                    error_msg = "STREAM READ FAILURE"
                    err_frame = np.zeros((480, 640, 3), np.uint8)
                    cv2.putText(err_frame, error_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    _, jpeg_err = cv2.imencode('.jpg', err_frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_err.tobytes() + b'\r\n')
                    break 
                time.sleep(0.1) 
                continue

            consecutive_read_errors = 0 
            frame_idx += 1
            current_interval_total_frames += 1 # Increment total frames processed for the interval
            processed_frame_bgr = frame_bgr.copy()
            
            current_frame_time = time.time()
            fps = 1 / (current_frame_time - last_frame_time) if (current_frame_time - last_frame_time) > 0 else 0
            current_interval_fps_values.append(fps)
            last_frame_time = current_frame_time

            if frame_idx % RECOGNITION_FRAME_INTERVAL == 0:
                try:
                    frame_rgb_pil = Image.fromarray(cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB))
                    boxes, probs = face_detector.detect(frame_rgb_pil)
                    
                    if boxes is not None and len(boxes) > 0:
                        current_interval_frames_with_faces += 1 
                        
                        for i, box in enumerate(boxes):
                            # FILTERING UNTUK DETEKSI WAJAH (MTCNN)
                            # Bounding box akan muncul jika confidence deteksi >= MIN_MTCNN_CONFIDENCE_STREAM (0.90)
                            if probs[i] < MIN_MTCNN_CONFIDENCE_STREAM: # Menggunakan konstanta yang sudah ada
                                continue 
                            
                            current_interval_total_faces_detected += 1 # Ini adalah TP Deteksi per interval
                            face_roi = get_face_roi_mtcnn(processed_frame_bgr, box)
                            if face_roi is None or face_roi.size == 0:
                                continue 

                            face_tensor = preprocess_face_for_resnet_pytorch(face_roi)
                            if face_tensor is None:
                                continue 

                            embedding = get_embedding_resnet_pytorch(face_tensor)
                            if embedding is None:
                                continue 
                            
                            recognized_id = "unknown"
                            recognized_name = "Unknown"
                            color = (0, 0, 255) # Default color for unknown/unrecognized
                            timer_text = ""
                            max_similarity = 0.0 

                            if known_embeddings_db["embeddings"].size > 0:
                                similarities = cosine_similarity(embedding.reshape(1, -1), known_embeddings_db["embeddings"])
                                best_idx = np.argmax(similarities[0])
                                max_similarity = similarities[0][best_idx]

                                # LOGIKA UTAMA UNTUK MENAMPILKAN NAMA:
                                # Nama hanya akan muncul jika max_similarity (confidence pengenalan) >= 0.7
                                # Dan juga memastikan personel berasal dari perusahaan yang login.
                                if max_similarity >= 0.6: # Gunakan 0.7 sesuai permintaan Anda untuk pengenalan
                                    recognized_id = known_embeddings_db["ids"][best_idx]
                                    personnel_info = personnel_map.get(str(recognized_id))
                                    
                                    if personnel_info:
                                        personnel_company_id = personnel_info.get("company_id")
                                        
                                        is_pegawai_valid = is_personnel_from_logged_in_company(personnel_company_id, current_user.id, db_session_obj)

                                        if is_pegawai_valid:
                                            # === Valid pegawai perusahaan login ===
                                            display_name = personnel_info.get("name", f"ID:{recognized_id}") # Nama muncul
                                            display_color = (0, 255, 0) # Hijau untuk dikenali
                                            current_interval_recognition_scores.append(max_similarity) 
                                            current_interval_total_faces_recognized += 1
                                            recognized_personnel_id = recognized_id

                                            # Timer logic...
                                            if cam_settings_obj and cam_settings_obj.role_camera == Camera_Settings.ROLE_TRACKING:
                                                now = datetime.now()
                                                if recognized_personnel_id in last_detection_time:
                                                    elapsed = (now - last_detection_time[recognized_personnel_id]).total_seconds()
                                                    detection_times[recognized_personnel_id] = detection_times.get(recognized_personnel_id, 0) + elapsed
                                                last_detection_time[recognized_personnel_id] = now

                                                timer_val = int(detection_times.get(recognized_personnel_id, 0))
                                                display_timer_text = f'Timer: {timer_val}s'

                                                if timer_val > 0 and (
                                                    recognized_personnel_id not in last_save_time_per_person or
                                                    (now - last_save_time_per_person[recognized_personnel_id]) >= timedelta(minutes=WORK_TIMER_SAVE_INTERVAL_MINUTES) # Menggunakan konstanta
                                                ):
                                                    try:
                                                        new_log = Work_Timer(
                                                            personnel_id=recognized_personnel_id,
                                                            camera_id=cam_settings_obj.id,
                                                            type=Work_Timer.TYPE_FACE_DETECTED,
                                                            datetime=datetime.utcnow(),
                                                            timer=timer_val
                                                        )
                                                        db_session_obj.add(new_log)
                                                        db_session_obj.commit()
                                                        last_save_time_per_person[recognized_personnel_id] = now
                                                        print(f"[WorkTimer] Saved for {display_name} ({timer_val}s)")
                                                    except Exception as db_ex:
                                                        db_session_obj.rollback()
                                                        print(f"[WorkTimer] DB Save Error for {recognized_personnel_id}: {db_ex}")
                                        else:
                                            # === Bukan pegawai perusahaan login atau similarity di bawah 0.7 ===
                                            # Nama akan tetap "Unknown" atau "Bukan Pegawai" jika tidak valid/cukup yakin
                                            print(f"[DEBUG] NOT SAME COMPANY → Mark as Bukan Pegawai")
                                            display_name = "Bukan Pegawai"
                                            display_color = (0, 0, 255) # Merah untuk tidak dikenali/bukan pegawai
                                            # current_interval_total_faces_unknown += 1
                                            current_interval_total_faces_false_positive += 1
                                            if recognized_id in last_detection_time:
                                                del last_detection_time[recognized_id]
                                            if recognized_id in detection_times:
                                                del detection_times[recognized_id]

                                    else:
                                        # === ID ditemukan tapi tidak di personnel_map (tidak sinkron atau corrupt) ===
                                        print(f"[DEBUG] ID {recognized_id} tidak ditemukan di personnel_map")
                                        display_name = "Unknown Personnel" # Nama akan Unknown
                                        display_color = (0, 0, 255) # Merah untuk tidak dikenali
                                        current_interval_total_faces_unknown += 1
                                        
                                        if recognized_id in last_detection_time:
                                            del last_detection_time[recognized_id]
                                        if recognized_id in detection_times:
                                            del detection_times[recognized_id]
                                else:
                                    display_name = "Unknown" # Nama akan Unknown jika similarity di bawah 0.7
                                    display_color = (0, 0, 255) # Merah untuk tidak dikenali
                                    current_interval_total_faces_unknown += 1
                                    
                                    # Hapus data timer jika sebelumnya dikenali tapi sekarang tidak cukup yakin
                                    if recognized_id in last_detection_time:
                                        del last_detection_time[recognized_id]
                                    if recognized_id in detection_times:
                                        del detection_times[recognized_id]
                            else: # No known embeddings to compare against
                                display_name = "Unknown"
                                display_color = (0, 0, 255)
                                current_interval_total_faces_unknown += 1

                            x1, y1, x2, y2 = [int(c) for c in box]
                            # Menentukan label yang akan ditampilkan di bounding box
                            if display_name in ["Bukan Pegawai", "Unknown", "Unknown Personnel"]:
                                label = display_name
                            else:
                                label = f"{display_name} ({max_similarity:.2f})"
                                if display_timer_text:
                                    label += f" {display_timer_text}"
                            label = label.strip()

                            cv2.rectangle(processed_frame_bgr, (x1, y1), (x2, y2), display_color, 2)
                            text_y = y1 - 10 if y1 > 20 else y1 + 20
                            cv2.putText(processed_frame_bgr, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

                except Exception as detect_rec_ex:
                    print(f"[WorkTimerStream {camera_source_url}] Detection/Recognition processing error: {detect_rec_ex}")
                    cv2.putText(processed_frame_bgr, "Processing Error", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            try:
                ret_encode, jpeg_bytes = cv2.imencode('.jpg', processed_frame_bgr)
                if ret_encode:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes.tobytes() + b'\r\n')
            except Exception as e_yield:
                print(f"[WorkTimerStream {camera_source_url}] Error yielding frame {frame_idx}: {e_yield}")
                break 

            time.sleep(0.001) 

    # except GeneratorExit:
    #     print(f"[WorkTimerStream {camera_source_url}] Client disconnected.")
    # except Exception as e_main:
    #     print(f"[WorkTimerStream {camera_source_url}] Unhandled error: {e_main}") # Added exc_info for full traceback
    # finally:
    #     if cap:
    #         release_camera_instance(camera_source_url)
    #     print(f"[WorkTimerStream {camera_source_url}] Stream ended.")

            # --- PERIODIC CSV LOGGING CHECK ---
            current_time = time.time()
            nama = "Saikem" # Default name if not recognized
            scenarios = ["frontal", "45deg", "samping", "terang", "redup", "kontras", "dekat", "sedang", "jauh", "kacamata", "masker", "webcam", "cctv"]
            if (current_time - last_log_time) >= LOG_INTERVAL_SECONDS:
                print(f"--- Logging performance for {camera_source_url} for the last {LOG_INTERVAL_SECONDS} seconds ---")

                avg_fps = np.mean(current_interval_fps_values) if current_interval_fps_values else 0.0

                # --- METRIK UNTUK FACE DETECTION (berbasis frame) ---
                scenario_type_fd = scenarios[12]
                # condition_variation_fd = "redup
                
                # Jml Percobaan: Total frames yang diproses dalam interval
                jml_percobaan_fd = current_interval_total_frames 
                
                # TP Deteksi: Jumlah frame di mana MTCNN berhasil mendeteksi setidaknya satu wajah
                true_positives_fd = current_interval_frames_with_faces
            
                # FN Deteksi: Jumlah frame yang diproses TAPI TIDAK ADA WAJAH terdeteksi oleh MTCNN.
                # Ini adalah proxy untuk "frame yang seharusnya terdeteksi tapi terlewat"
                false_negatives_fd = current_interval_total_frames - current_interval_frames_with_faces

                if (true_positives_fd + false_negatives_fd) > 0:
                    tpr_fd = (true_positives_fd / (true_positives_fd + false_negatives_fd)) * 100
                    fnr_fd = (false_negatives_fd / (true_positives_fd + false_negatives_fd)) * 100
                else:
                    tpr_fd = 0.0
                    fnr_fd = 0.0
                    
                notes_fd = f"Deteksi wajah oleh MTCNN dari total {current_interval_total_frames} frame. Dideteksi di {current_interval_frames_with_faces} frame."
                
                log_face_detection_metrics_to_csv(
                    nama,
                    scenario_type_fd,
                    avg_fps,
                    jml_percobaan_fd,
                    true_positives_fd,
                    false_negatives_fd,
                    tpr_fd,
                    fnr_fd,
                    notes_fd
                )


                
                # scenario_type_fr = scenarios[8]
                condition_variation_fr = f"(Avg. FPS: {avg_fps:.2f})"
                
                # Jml Percobaan: Total wajah yang *terdeteksi* oleh MTCNN dan kemudian mencoba dikenali
                jml_percobaan_fr = current_interval_total_faces_detected
                
                # TP Recognition: Jumlah wajah yang terdeteksi DAN berhasil dikenali
                true_positives_fr = current_interval_total_faces_recognized
                
                # FN Recognition: Jumlah wajah yang terdeteksi TAPI TIDAK dikenali (unknown)
                false_negatives_fr = current_interval_total_faces_unknown 
                false_positive_fr = current_interval_total_faces_false_positive 

                if jml_percobaan_fr > 0:
                    tpr_fr = (true_positives_fr / jml_percobaan_fr) * 100
                    fnr_fr = (false_negatives_fr / jml_percobaan_fr) * 100
                else:
                    tpr_fr = 0.0
                    fnr_fr = 0.0
                
                notes_fr = f"Pengenalan wajah dari {jml_percobaan_fr} wajah yang terdeteksi. "
                if current_interval_recognition_scores:
                    notes_fr += f"Rata-rata Confidence: {np.mean(current_interval_recognition_scores):.2f}."
                
                log_face_recognition_metrics_to_csv(
                    nama,
                    scenario_type_fd,
                    jml_percobaan_fr,
                    true_positives_fr,
                    false_positive_fr,
                    false_negatives_fr,
                    tpr_fr,
                    fnr_fr,
                    np.mean(current_interval_recognition_scores) if current_interval_recognition_scores else 0.0,
                    notes_fr
                )

                
                current_interval_total_frames = 0
                current_interval_frames_with_faces = 0
                current_interval_total_faces_detected = 0
                current_interval_total_faces_recognized = 0
                current_interval_total_faces_unknown = 0
                last_log_time = current_time
                current_interval_recognition_scores = []
                current_interval_fps_values = [] 

    except GeneratorExit:
        print(f"[WorkTimerStream {camera_source_url}] Client disconnected.")
    except Exception as e_main:
        print(f"[WorkTimerStream {camera_source_url}] Unhandled error: {e_main}") # Added exc_info for full traceback
    finally:
        if cap:
            release_camera_instance(camera_source_url)
        print(f"[WorkTimerStream {camera_source_url}] Stream ended.")

# ... (rest of your dummy classes and __main__ block) ...
def log_accuracy_to_csv(
    camera_url,
    recognized_faces,
    recognized_percent,
    unknown_faces,
    unknown_percent,
    # avg_confidence
    
):
    """Saves accuracy metrics to a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(CSV_LOG_DIR, f"work_timer_accuracy_{timestamp}.csv")

    # Check if file exists to write header only once
    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([
                "Timestamp", "Camera URL",
                "Total Frames Processed", "Frames with Face Detected", "Frame Detection Accuracy (%)",
                "Recognized Faces", "Recognized Faces (%)",
                "Unknown Faces", "Unknown Faces (%)"
            ])
        
        writer.writerow([
            timestamp, camera_url,
            recognized_faces, f"{recognized_percent:.2f}",
            unknown_faces, f"{unknown_percent:.2f}"
        ])
    print(f"Accuracy metrics saved to {file_path}")
    
    
# --- Lokasi File CSV (Bisa dikonfigurasi di luar fungsi atau sebagai parameter) ---
CSV_LOG_PATH = 'presence_log_streaming.csv'
# Headers untuk CSV
CSV_HEADERS = ['timestamp_log', 'nama_pegawai', 'status_presensi', 'selisih_waktu_deteksi_db_ms', 'confidence_deteksi', 'deteksi_pertama_kali', 'waktu_simpan_db_selesai', 'keberhasilan']

# Fungsi untuk menulis ke CSV
def append_to_csv(data_row, filename=CSV_LOG_PATH, headers=CSV_HEADERS):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(data_row)
    except IOError as e:
        print(f"ERROR: Could not write to CSV file {filename}: {e}")
    
# ====================================================================
# Streaming for PRESENCE (Attendance Taking with SSE)
# ====================================================================
def generate_presence_stream_frames(app_for_context, cam_id_db): # cam_id_db is the DB ID of Camera_Settings
    global face_detector, resnet_embedder, known_embeddings_db, device
    global last_presence_attempt_time_personnel, presence_message_queue

    logger = app_for_context.logger
    logger.info(f"[PresenceStream DB_ID:{cam_id_db}] Starting stream.")
    
    # Tambahkan inisialisasi variabel interval di sini
    current_interval_total_frames = 0
    current_interval_frames_with_faces = 0
    current_interval_total_faces_detected = 0
    current_interval_total_faces_recognized = 0
    current_interval_total_faces_unknown = 0

    camera_setting = None
    personnel_map = {}
    camera_feed_url = None

    # --- Load Config Constants (scoped to this generator) ---
    MIN_MTCNN_CONF_PRESENCE = get_config_val('MIN_MTCNN_CONFIDENCE_STREAM', 0.90, float) # Reuse stream conf or define specific
    RECOG_PRESENCE_INTERVAL_S = get_config_val('RECOGNITION_AND_PRESENCE_INTERVAL_SECONDS', 3, int)
    COSINE_SIM_THRESH_PRESENCE = get_config_val('COSINE_SIMILARITY_THRESHOLD', 0.80, float)
    MIN_S_BETWEEN_PRESENCE_PERSONNEL = get_config_val('MIN_SECONDS_BETWEEN_PRESENCE_ATTEMPTS_PERSONNEL', 20, int)
    BASE_UPLOAD_FOLDER_PROOFS = get_upload_folder_presence_proofs() # Get the correct path


    try:
        camera_setting = db.session.query(Camera_Settings).get(cam_id_db) # Use db.session if available
        if not camera_setting or not camera_setting.cam_is_active:
            error_msg = f"CAMERA DB_ID {cam_id_db} INACTIVE/NOT FOUND"
            print(f"[PresenceStream DB_ID:{cam_id_db}] {error_msg}")
            frame = np.zeros((480,640,3),np.uint8); cv2.putText(frame,error_msg,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2); _,jpeg=cv2.imencode('.jpg',frame);yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n'); return
        
        camera_feed_url = camera_setting.feed_src
        logger.info(f"[PresenceStream DB_ID:{cam_id_db}] Using feed URL: {camera_feed_url}")

        all_personnels = db.session.query(Personnels).all()
        personnel_map = {str(p.id): p.name for p in all_personnels}
        logger.info(f"[PresenceStream DB_ID:{cam_id_db}] Personnel map loaded ({len(personnel_map)} entries).")

    except Exception as e_init:
        print(f"[PresenceStream DB_ID:{cam_id_db}] Error fetching initial data: {e_init}", exc_info=True)
        # ... (yield error frame and return) ...
        error_msg = "ERROR DATA AWAL"; frame = np.zeros((480,640,3),np.uint8); cv2.putText(frame,error_msg,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2); _,jpeg=cv2.imencode('.jpg',frame);yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n'); return


    # --- AI Model & Embeddings Checks ---
    if face_detector is None or resnet_embedder is None:
        error_msg = "AI MODEL NOT READY"
        print(f"[PresenceStream DB_ID:{cam_id_db}] {error_msg}")
        frame = np.zeros((480,640,3),np.uint8); cv2.putText(frame,error_msg,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2); _,jpeg=cv2.imencode('.jpg',frame);yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n'); return

    if not known_embeddings_db or "embeddings" not in known_embeddings_db or \
       not isinstance(known_embeddings_db["embeddings"], np.ndarray) or \
       known_embeddings_db["embeddings"].size == 0 or not known_embeddings_db["ids"]:
        error_msg = "EMBEDDINGS DB EMPTY"
        logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] {error_msg}")
        frame = np.zeros((480,640,3),np.uint8); cv2.putText(frame,error_msg,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,128,255),2); _,jpeg=cv2.imencode('.jpg',frame);yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n'); return

    # --- Camera Initialization ---
    cap = get_camera_instance(camera_feed_url)
    if not cap or not cap.isOpened():
        error_msg = f"CAMERA ERROR: {camera_feed_url}"
        print(f"[PresenceStream DB_ID:{cam_id_db}] {error_msg}")
        frame = np.zeros((480,640,3),np.uint8); cv2.putText(frame,error_msg,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2); _,jpeg=cv2.imencode('.jpg',frame);yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n'); return

    last_recognition_trigger_time = time.time()
    frame_idx_presence = 0 
    # consecutive_read_errors_presence = 0
    MAX_CONSECUTIVE_READ_ERRORS_PRESENCE = 60

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                # consecutive_read_errors_presence += 1
                # logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] Frame read failed ({consecutive_read_errors_presence}/{MAX_CONSECUTIVE_READ_ERRORS_PRESENCE}).")
                if consecutive_read_errors_presence > MAX_CONSECUTIVE_READ_ERRORS_PRESENCE:
                    print(f"[PresenceStream DB_ID:{cam_id_db}] Max read errors. Stopping.")
                    error_msg = "STREAM READ FAILURE"; err_frame = np.zeros((480,640,3),np.uint8); cv2.putText(err_frame,error_msg,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2); _,jpeg_err=cv2.imencode('.jpg',err_frame); yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg_err.tobytes()+b'\r\n'); break
                time.sleep(0.1)
                continue
            
            consecutive_read_errors_presence = 0
            if frame_bgr is None or frame_bgr.size == 0:
                logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] Frame {frame_idx_presence} empty. Skipping.")
                continue
            
            # current_interval_total_frames += 1
            frame_idx_presence += 1
            processed_frame_bgr = frame_bgr.copy() # For drawing
            current_time_epoch = time.time()
            time_face_detection_start = None

            # Perform recognition and presence attempt at specified interval
            if (current_time_epoch - last_recognition_trigger_time) >= RECOG_PRESENCE_INTERVAL_S:
                last_recognition_trigger_time = current_time_epoch
                
                # Basic frame validation before AI
                if not isinstance(processed_frame_bgr, np.ndarray) or \
                   processed_frame_bgr.dtype != np.uint8 or \
                   processed_frame_bgr.ndim != 3 or processed_frame_bgr.shape[2] != 3:
                    logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] Invalid frame type/dims for AI. Skipping this cycle.")
                else:
                    frame_h_val, frame_w_val = processed_frame_bgr.shape[:2]
                    min_dim_presence = 32
                    if frame_h_val < min_dim_presence or frame_w_val < min_dim_presence:
                        logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] Frame too small for AI. Skipping this cycle.")
                    else:
                        # if frame_idx_presence % RECOGNITION_FRAME_INTERVAL == 0:
                        try:
                            time_face_detection_start = time.time()
                            frame_rgb_pil_presence = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)) # Use original frame_bgr for detection
                            boxes, probs = face_detector.detect(frame_rgb_pil_presence)

                            if boxes is not None:
                                # current_interval_frames_with_faces += 1
                                for i, box in enumerate(boxes):
                                    confidence = probs[i]
                                    if confidence < MIN_MTCNN_CONF_PRESENCE:
                                        continue

                                    try:
                                        box = np.array(box, dtype=np.float32)
                                        if np.any(np.isnan(box)) or np.any(np.isinf(box)) or box[2]-box[0]<=0 or box[3]-box[1]<=0:
                                            logger.debug(f"[PresenceStream DB_ID:{cam_id_db}] Invalid MTCNN box. Skipping.")
                                            continue
                                    except Exception as e_box:
                                        logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] Error converting box: {e_box}")
                                        continue


                                    face_roi = get_face_roi_mtcnn(frame_bgr, box) # Get ROI from original BGR
                                    if face_roi is None or face_roi.size == 0: continue
                                    
                                    face_tensor = preprocess_face_for_resnet_pytorch(face_roi)
                                    if face_tensor is None: continue
                                    embedding = get_embedding_resnet_pytorch(face_tensor)
                                    if embedding is None: continue

                                    recognized_name = "Unknown"; max_similarity = 0.0; recognized_personnel_id_int = None

                                    if known_embeddings_db["embeddings"].size > 0:
                                        similarities = cosine_similarity(embedding.reshape(1, -1), known_embeddings_db["embeddings"])
                                        best_match_idx = np.argmax(similarities[0])
                                        max_similarity = similarities[0][best_match_idx]

                                        if max_similarity >= 0.75:
                                            try:
                                                potential_id_str = known_embeddings_db["ids"][best_match_idx]
                                                recognized_personnel_id_int = int(potential_id_str)
                                                recognized_name = personnel_map.get(str(recognized_personnel_id_int), f"ID:{recognized_personnel_id_int}")
                                            except (ValueError, IndexError) as e_id:
                                                print(f"[PresenceStream DB_ID:{cam_id_db}] Error parsing personnel ID from DB: {potential_id_str if 'potential_id_str' in locals() else 'UNKNOWN'} - {e_id}")
                                                recognized_personnel_id_int = None; recognized_name = "Error: Invalid ID"
                                    
                                    # --- Process Attendance if recognized ---
                                    if recognized_personnel_id_int:
                                        now_dt = datetime.now()
                                        can_attempt = True
                                        if recognized_personnel_id_int in last_presence_attempt_time_personnel:
                                            if (now_dt - last_presence_attempt_time_personnel[recognized_personnel_id_int]).total_seconds() < MIN_S_BETWEEN_PRESENCE_PERSONNEL:
                                                can_attempt = False
                                        
                                        if can_attempt:
                                            last_presence_attempt_time_personnel[recognized_personnel_id_int] = now_dt
                                            saved_proof_path = None # Relative path for DB

                                            # Save proof image (using face_roi, which is BGR)
                                            try:
                                                proof_subfolder = now_dt.strftime('%Y%m%d')
                                                save_dir_abs = os.path.join(BASE_UPLOAD_FOLDER_PROOFS, proof_subfolder)
                                                os.makedirs(save_dir_abs, exist_ok=True)
                                                
                                                clean_rec_name_proof = clean_name_for_filename(recognized_name)
                                                proof_filename = f"{recognized_personnel_id_int}_{clean_rec_name_proof}_{now_dt.strftime('%H%M%S%f')}_{int(max_similarity*100)}.jpg"
                                                proof_filepath_abs = os.path.join(save_dir_abs, proof_filename)
                                                
                                                if cv2.imwrite(proof_filepath_abs, face_roi):

                                                    static_folder_abs = app_for_context.static_folder
                                                    saved_proof_path = os.path.relpath(proof_filepath_abs, static_folder_abs).replace("\\", "/")
                                                    logger.info(f"[PresenceStream DB_ID:{cam_id_db}] Proof for {recognized_name} saved: {saved_proof_path}")
                                                else:
                                                    print(f"[PresenceStream DB_ID:{cam_id_db}] Failed to write proof image for {recognized_name} to {proof_filepath_abs}")
                                            except Exception as e_img_save:
                                                print(f"[PresenceStream DB_ID:{cam_id_db}] Error saving proof image for {recognized_name}: {e_img_save}", exc_info=True)

                                            attendance_payload = {
                                                'personnel_id': recognized_personnel_id_int, 'name': recognized_name,
                                                'datetime': now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                                                'image_path': saved_proof_path, 'camera_id': cam_id_db,
                                                'confidence': float(max_similarity),
                                                'detection_time_start': time_face_detection_start # Waktu deteksi pertama kali
                                            }

                                            # Melewatkan logger ke process_attendance_entry
                                            attendance_status_code = process_attendance_entry(attendance_payload) # FIX: Lewatkan logger
                                            
                                            message_to_client = ""
                                            message_type = "info"

                                            if attendance_status_code == 'success':
                                                # Menangani kasus 'success' secara eksplisit untuk ONTIME, LATE, dan LEAVE.
                                                # Ini memperbaiki bug utama di mana notifikasi sukses dianggap error.
                                                message_to_client = f"Absensi untuk {recognized_name} berhasil dicatat."
                                                message_type = "success"
                                            elif attendance_status_code == 'already_attended_on_time_or_late':
                                                message_to_client = f"{recognized_name} sudah absen masuk hari ini."
                                                message_type = "warning"
                                            elif attendance_status_code == 'already_left_today':
                                                message_to_client = f"{recognized_name} sudah absen pulang hari ini."
                                                message_type = "warning"
                                            elif attendance_status_code == 'cannot_leave_before_attendance':
                                                message_to_client = f"{recognized_name} belum tercatat absen masuk hari ini."
                                                message_type = "warning"
                                            elif attendance_status_code in ['ignored_unknown_time_slot', 'out_of_time']:
                                                message_to_client = f"Absen {recognized_name} di luar jam kerja yang ditentukan."
                                                message_type = "info"
                                            else:  # Menangani semua kode error lain seperti 'db_error_*', 'input_error_*', dll.
                                                message_to_client = f"Gagal memproses absensi. Bukan Pegawai "
                                                message_type = "error"
                                            
                                            # Put to queue (this part is likely fine)
                                            try:
                                                presence_message_queue.put_nowait({
                                                    'type': message_type,
                                                    'message': message_to_client,
                                                    'personnel_name': recognized_name,
                                                    'timestamp': now_dt.strftime('%H:%M:%S'),
                                                    'status_code': attendance_status_code
                                                })
                                                print(presence_message_queue)
                                            except queue.Full:
                                                logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] SSE queue full. Message for {recognized_name} dropped.")
                                            except Exception as e_q_put:
                                                print(f"[PresenceStream DB_ID:{cam_id_db}] Error putting message to SSE queue: {e_q_put}", exc_info=True)
                                        else: # Cooldown active
                                            logger.debug(f"[PresenceStream DB_ID:{cam_id_db}] Presence attempt for {recognized_name} (ID: {recognized_personnel_id_int}) skipped due to cooldown.")


                                    # Draw on processed_frame_bgr for display
                                    x1_disp, y1_disp, x2_disp, y2_disp = [int(c) for c in box]
                                    disp_color = (0,255,0) if recognized_personnel_id_int else (0,0,255)
                                    cv2.rectangle(processed_frame_bgr, (x1_disp,y1_disp), (x2_disp,y2_disp), disp_color, 2)
                                    text = f"{recognized_name} ({max_similarity:.2f})"
                                    cv2.putText(processed_frame_bgr, text, (x1_disp, y1_disp - 10 if y1_disp > 20 else y1_disp + 20), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, disp_color, 1) # Smaller font for presence
                        except cv2.error as cv_err_presence:
                            print(f"[PresenceStream DB_ID:{cam_id_db}] OpenCV error in AI cycle: {cv_err_presence}", exc_info=False)
                            cv2.putText(processed_frame_bgr, "AI CV2 Err", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        except Exception as e_presence_ai:
                            print(f"[PresenceStream DB_ID:{cam_id_db}] Generic error in AI cycle: {e_presence_ai}", exc_info=True)
                            cv2.putText(processed_frame_bgr, "AI Proc Err", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
            # Encode and yield the frame (even if AI part was skipped this cycle)
            try:
                ret_enc, jpeg_bytes_disp = cv2.imencode('.jpg', processed_frame_bgr)
                if not ret_enc:
                    # logger.warning(f"[PresenceStream DB_ID:{cam_id_db}] JPEG encode failed frame {frame_idx_presence}.")
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_bytes_disp.tobytes() + b'\r\n')
            except Exception as e_yield_presence:
                # print(f"[PresenceStream DB_ID:{cam_id_db}] Error yielding frame: {e_yield_presence}", exc_info=True)
                break
            
            time.sleep(0.01) # Short sleep for responsiveness

    except GeneratorExit:
        logger.info(f"[PresenceStream DB_ID:{cam_id_db}] Client disconnected.")
    except Exception as e_main_loop_presence:
        print(f"[PresenceStream DB_ID:{cam_id_db}] Unhandled Main Loop Exception: {e_main_loop_presence}", exc_info=True)
        # ... (yield error frame) ...
    finally:
        logger.info(f"[PresenceStream DB_ID:{cam_id_db}] Exiting. Releasing camera URL: {camera_feed_url if camera_feed_url else 'N/A'}.")
        if cap: release_camera_instance(camera_feed_url)


# ====================================================================
# Routes (Using MTCNN and Separated Stream Generators)
# ====================================================================

# --- Dataset Management Routes (capture_page, capture_faces_route, train_model_route, dataset_no_id, delete_images) ---
# These will now use MTCNN via face_detector global and MIN_MTCNN_CONFIDENCE_DATASET
@bp.route('/capture')
@login_required
@employee_required # Ensure this decorator exists and works
def capture_page():
    personnel = Personnels.query.filter_by(user_id=current_user.id).first() # Match by user_id
    if not personnel:
        flash("Personnel data not found for your account. Contact admin.", "danger")
        return redirect(url_for('main.dashboard')) # Adjust redirect as needed
    return render_template('face/capture.html', name=personnel.name, personnel_id=personnel.id) # Ensure template path

@bp.route('/capture_data', methods=['POST'])
@login_required
@employee_required
def capture_faces_route():
    personnel = Personnels.query.filter_by(user_id=current_user.id).first()
    if not personnel: return jsonify({'status': 'error', 'message': 'User personnel data not found.'}), 400
    
    if face_detector is None: # Check global MTCNN detector
        # current_app.print("Face detector (MTCNN) not ready for /capture_data.")
        return jsonify({'status': 'error', 'message': 'Face detector not ready.'}), 500
    
    result = capture_faces_logic(personnel) # This now uses MTCNN
    
    if result.get('success'):
        current_app.logger.info("New face data captured, initiating model retraining...")
        train_result = _train_face_model_logic() # Panggil fungsi training
        if train_result.get('success'):
            flash(f"{result.get('message', 'Capture process ended.')} Model training updated: {train_result.get('message')}", 'success')
        else:
            flash(f"{result.get('message', 'Capture process ended.')} Model training failed: {train_result.get('message')}", 'warning')
    else:
        flash(result.get('message', 'Capture process ended.'), result.get('status', 'info'))
        
    return jsonify(result)

@bp.route('/train', methods=['POST'])
# @login_required
def train_model_route():
    if face_detector is None or resnet_embedder is None:
        # current_app.print("AI models (MTCNN/ResNet) not ready for /train.")
        return jsonify({'status': 'error', 'message': 'AI models not ready. Check server logs.'}), 500

    result = _train_face_model_logic() # This now uses MTCNN
    return jsonify({
        'status': 'success' if result.get('success') else 'error',
        'message': result.get('message', 'Training process ended.')
    }), 200 if result.get('success') else 500
    
@bp.route('/dataset') # Route for listing own dataset or selecting for admin
@bp.route('/dataset/<int:personnel_id>') # Route for admin viewing specific dataset
@login_required
def dataset_no_id(personnel_id=None):
    # ... (Your existing dataset viewing logic, ensure it uses get_personnel_folder_path_local correctly) ...
    # This part doesn't directly involve AI models, just file listing.
    # Make sure clean_name_for_filename is used consistently if folder names depend on it.
    current_personnel_viewing = None
    is_own_profile = False

    if current_user.has_role('admin') or current_user.has_role('superadmin'):
        if personnel_id:
            current_personnel_viewing = Personnels.query.get(personnel_id)
            if not current_personnel_viewing:
                flash("Personnel ID not found.", "danger")
                return redirect(url_for('admin.manage_personnels')) # Adjust as needed
            # Optional: company check for admin
            if hasattr(current_user, 'company_id') and hasattr(current_personnel_viewing, 'company_id') and \
               current_user.company_id != current_personnel_viewing.company_id and not current_user.has_role('superadmin'):
                flash("Unauthorized to view this personnel's dataset.", "danger")
                return redirect(url_for('admin.manage_personnels'))
        else: # Admin needs to select a personnel from a list usually
            flash("Please select a personnel to view their dataset.", "info")
            return redirect(url_for('admin.manage_personnels')) # Or a page that lists personnel
    elif current_user.has_role('employee'):
        user_personnel = Personnels.query.filter_by(user_id=current_user.id).first()
        if not user_personnel:
            flash("Your personnel profile was not found.", "danger")
            return redirect(url_for('employee.dashboard')) # Adjust
        if personnel_id and personnel_id != user_personnel.id:
            flash("You can only view your own dataset.", "warning")
            # Fallback to own dataset or redirect
            current_personnel_viewing = user_personnel
            is_own_profile = True
        else:
            current_personnel_viewing = user_personnel
            is_own_profile = True
    else: # Other roles or unhandled
        flash("Unauthorized to view datasets.", "danger")
        return redirect(url_for('main.index')) # Adjust

    if not current_personnel_viewing: # Should be caught above, but as a safeguard
        flash("Personnel could not be determined for dataset view.", "danger")
        return redirect(url_for('main.index'))

    images = []
    personnel_name_folder_safe = clean_name_for_filename(current_personnel_viewing.name)
    personnel_folder_abs = os.path.join(get_personnel_folder_path_local(), personnel_name_folder_safe)
    
    if os.path.exists(personnel_folder_abs):
        for file_name in sorted(os.listdir(personnel_folder_abs)): # Sort for consistency
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Path relative to Flask's static folder for url_for
                    # Assumes get_personnel_folder_path_local() is <app_root>/app/static/img/personnel_pics
                    # and static_folder is <app_root>/app/static
                    path_inside_static_img_folder = os.path.join('img', 'personnel_pics', personnel_name_folder_safe, file_name).replace("\\", "/")
                    images.append({
                        'url': url_for('static', filename=path_inside_static_img_folder),
                        'filename': file_name,
                        'personnel_name': current_personnel_viewing.name 
                    })
                except Exception as e_url:
                    print(f"Error creating URL for {file_name}: {e_url}")
    else:
        flash(f"No dataset images found for {current_personnel_viewing.name}.", "info")
        
    return render_template('face/dataset.html', # Ensure template path
                           images=images, 
                           name=current_personnel_viewing.name, 
                           personnel=current_personnel_viewing,
                           is_own_profile=is_own_profile)


@bp.route('/delete_images', methods=['POST'])
@login_required
def delete_images():
    # ... (Your existing delete logic, ensure it uses clean_name_for_filename for folder path) ...
    # This part also doesn't directly involve AI models, just file and DB operations.
    # Remember to flash a message to re-train embeddings after deletion.
    if request.method == 'POST':
        images_to_delete = request.form.getlist('images_to_delete')
        personnel_id_from_form = request.form.get('personnel_id') # Use ID for safety

        if not personnel_id_from_form:
            flash("Personnel ID missing from form.", "danger")
            return redirect(request.referrer or url_for('main.index'))
        
        try:
            personnel_id_int = int(personnel_id_from_form)
        except ValueError:
            flash("Invalid Personnel ID format.", "danger")
            return redirect(request.referrer or url_for('main.index'))

        personnel_to_edit = Personnels.query.get(personnel_id_int)
        if not personnel_to_edit:
            flash("Personnel not found.", "danger")
            return redirect(request.referrer or url_for('main.index'))
        
        # Authorization
        can_delete = False
        if current_user.has_role('admin') or current_user.has_role('superadmin'):
            # Admin/Superadmin can delete if same company or superadmin
            if hasattr(current_user, 'company_id') and hasattr(personnel_to_edit, 'company_id') and \
               current_user.company_id == personnel_to_edit.company_id or current_user.has_role('superadmin'):
                can_delete = True
        elif current_user.has_role('employee') and personnel_to_edit.user_id == current_user.id:
            can_delete = True # Employee can delete their own

        if not can_delete:
            flash("Unauthorized to delete these images.", "danger")
            return redirect(request.referrer or url_for('face.dataset_no_id', personnel_id=personnel_to_edit.id))

        deleted_count = 0
        personnel_name_folder_safe = clean_name_for_filename(personnel_to_edit.name)
        personnel_folder_abs = os.path.join(get_personnel_folder_path_local(), personnel_name_folder_safe)

        for filename in images_to_delete:
            if not filename or ".." in filename or "/" in filename or "\\" in filename: # Basic security
                flash(f"Invalid filename detected: {filename}", "warning")
                continue

            full_path_abs = os.path.join(personnel_folder_abs, filename)
            if os.path.exists(full_path_abs) and os.path.isfile(full_path_abs): # Ensure it's a file
                try:
                    os.remove(full_path_abs)
                    # Delete from Personnel_Images DB
                    # Path stored in DB is relative to 'static/img/' or similar base for UPLOAD_FOLDER
                    # Example: 'personnel_pics/John_Doe/face_1_John_Doe_001.jpg'
                    db_image_path_rel_to_personnel_pics = os.path.join(personnel_name_folder_safe, filename).replace("\\", "/")
                    path_for_db_query = os.path.join('personnel_pics', db_image_path_rel_to_personnel_pics).replace("\\","/")

                    img_db_entry = Personnel_Images.query.filter_by(personnel_id=personnel_to_edit.id, image_path=path_for_db_query).first()
                    if img_db_entry:
                        db.session.delete(img_db_entry)
                    deleted_count += 1
                except Exception as e:
                    flash(f"Error deleting {filename}: {e}", "warning")
            else:
                flash(f"File not found or invalid: {filename}", "warning")
        
        if deleted_count > 0:
            try:
                db.session.commit()
                flash(f"Successfully deleted {deleted_count} images for {personnel_to_edit.name}. Please re-run 'Train Model' to update face recognition data.", "success")
            except Exception as e_commit:
                db.session.rollback()
                flash(f"Database error after deleting images: {e_commit}", "danger")
        else:
            flash("No images were selected or deleted.", "info")
            
        return redirect(url_for('face.dataset_no_id', personnel_id=personnel_to_edit.id))
    return jsonify({'status': 'error', 'message': 'Invalid request method.'}), 405


# --- Camera Instance Management (get_camera_instance, release_camera_instance) ---
# This logic can remain largely the same.
def get_camera_instance(camera_source):
    global _camera_instance_cache
    logger = current_app.logger if current_app else print

    processed_source = int(camera_source) if isinstance(camera_source, str) and camera_source.isdigit() else camera_source
    cache_key = str(processed_source)

    if cache_key in _camera_instance_cache:
        cap = _camera_instance_cache[cache_key]
        if cap.isOpened():
            return cap
        else:
            cap.release()
            del _camera_instance_cache[cache_key]

    cap_new = None
    tried_backends_log = []

    logger.info(f"[get_camera_instance] Attempting to open new camera instance for source: {processed_source}")
    
    # === Untuk Kamera Lokal ===
    if isinstance(processed_source, int):
        backends_to_try = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in backends_to_try:
            tried_backends_log.append(str(backend))
            cap_new = cv2.VideoCapture(processed_source, backend)
            if cap_new.isOpened():
                cap_new.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"[get_camera_instance] Local camera {processed_source} opened with backend {backend}.")
                break
            else:
                cap_new.release()
                cap_new = None
    else:
        # === Untuk RTSP ===
        backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        for backend in backends_to_try:
            tried_backends_log.append(str(backend))
            cap_new = cv2.VideoCapture(str(processed_source), backend)

            if cap_new.isOpened():
                # Hindari delay dengan buffer minimum
                cap_new.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # # Kadang perlu set resolusi lebih rendah untuk percepat proses
                cap_new.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_new.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

                logger.info(f"[get_camera_instance] Stream {processed_source} opened with backend {backend}.")
                break
            else:
                cap_new.release()
                cap_new = None

    if cap_new and cap_new.isOpened():
        _camera_instance_cache[cache_key] = cap_new
        return cap_new
    else:
        print(f"[get_camera_instance] FAILED to open camera source '{processed_source}'. Tried backends: {tried_backends_log}")
        return None


def release_camera_instance(camera_source): # camera_source is the original identifier (int or URL string)
    global _camera_instance_cache
    logger = current_app.logger if current_app else print
    cache_key = str(int(camera_source) if isinstance(camera_source, str) and camera_source.isdigit() else camera_source)

    if cache_key in _camera_instance_cache:
        cap = _camera_instance_cache[cache_key]
        if cap.isOpened():
            logger.info(f"Releasing camera instance for source '{cache_key}' from cache.")
            cap.release()
        del _camera_instance_cache[cache_key]
    else:
        logger.debug(f"No cached camera instance found to release for source '{cache_key}'.")


# --- Streaming Routes (Using new MTCNN based generators) ---
@bp.route('/recognize_stream/<path:cam_id>') 
@login_required
def recognize_stream_route(cam_id):

    logger = current_app.logger
    cam_settings = None
    
    actual_camera_feed_url = None
    cam_id_for_settings = None

    if cam_id:
        cam_id_int = int(cam_id)
        if cam_id_int > 1:
            cam_settings = Camera_Settings.query.get(cam_id_int)
            if cam_settings:
                if cam_settings.cam_is_active:
                    actual_camera_feed_url = cam_settings.feed_src
                    cam_id_for_settings = cam_settings.id
                    logger.info(f"Recognize Stream: Using camera '{cam_settings.cam_name}' (ID: {cam_id_for_settings}) with feed: {actual_camera_feed_url}")
                else:
                    flash(f"Camera '{cam_settings.cam_name}' (ID: {cam_id_int}) is inactive.", "warning")
                    return redirect(request.referrer or url_for('main.index'))
            else:
                if cam_id_int in [0, 1, 2, 3]: 
                     actual_camera_feed_url = cam_id_int
                     logger.info(f"Recognize Stream: Using direct camera index: {actual_camera_feed_url}")
                else:
                     flash(f"Camera with DB ID {cam_id_int} not found, and it's not a typical cctv index.", "danger")
                     return redirect(request.referrer or url_for('main.index'))
        else: 
            actual_camera_feed_url = cam_id_int
            logger.info(f"Recognize Stream: Using direct camera index: {actual_camera_feed_url}")
            cam_settings = Camera_Settings.query.filter_by(feed_src=str(cam_id_int), cam_is_active=True).first()
            if cam_settings:
                cam_id_for_settings = cam_settings.id
                logger.info(f"Recognize Stream: Found matching DB settings for index {cam_id_int}: Cam '{cam_settings.cam_name}' (ID: {cam_id_for_settings})")

    else: 
        actual_camera_feed_url = cam_id
        logger.info(f"Recognize Stream: Using direct camera URL: {actual_camera_feed_url}")
        cam_settings = Camera_Settings.query.filter_by(feed_src=actual_camera_feed_url, cam_is_active=True).first()
        if cam_settings:
            cam_id_for_settings = cam_settings.id
            logger.info(f"Recognize Stream: Found matching DB settings for URL: Cam '{cam_settings.cam_name}' (ID: {cam_id_for_settings})")

    if actual_camera_feed_url is None: # Should not happen if logic above is correct
        flash("Could not determine camera source.", "danger")
        return redirect(request.referrer or url_for('main.index'))

    if face_detector is None or resnet_embedder is None: 
        flash("AI Models (MTCNN/ResNet) not ready. Contact administrator.", "danger")
        print("Recognize Stream: AI models not initialized when route accessed.")
        return redirect(request.referrer or url_for('main.index'))

    app_ctx_obj = current_app._get_current_object()

    return Response(
        stream_with_context(generate_work_timer_stream_frames(actual_camera_feed_url, cam_settings, app_ctx_obj, db.session)),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@bp.route('/live_presence_feed/<int:cam_id>') # Uses DB ID of camera_settings
@login_required
def live_presence_feed(cam_id):

    logger = current_app.logger
    logger.info(f"Route /live_presence_feed: DB Camera ID: {cam_id}, User: {current_user.username if current_user else 'Anonymous'}")


    if face_detector is None or resnet_embedder is None:
        logger.warning("Presence Stream: AI models not ready, attempting initialization from route.")
        with current_app.app_context():
            initialize_ai_models(current_app) 
        if face_detector is None or resnet_embedder is None:
            print("Presence Stream: AI models FAILED to initialize. Aborting stream.")
            error_frame_html = "<html><body><h1>AI System Offline</h1></body></html>" 
            return Response(error_frame_html, status=503, mimetype='text/html')


    app_for_gen = current_app._get_current_object()

    def generate_wrapper_presence(app_obj_wrapper, camera_id_for_gen):

        logger.info(f"Presence Wrapper: Starting generate_presence_stream_frames for DB Cam ID: {camera_id_for_gen}")
        for frame_bytes in generate_presence_stream_frames(app_obj_wrapper, camera_id_for_gen):
            yield frame_bytes
        logger.info(f"Presence Wrapper: Finished generate_presence_stream_frames for DB Cam ID: {camera_id_for_gen}")

    return Response(stream_with_context(generate_wrapper_presence(app_for_gen, cam_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Route and Page for Presence Stream UI ---
@bp.route('/presence_stream_page/<int:cam_id>')
@login_required
def presence_stream_page(cam_id):
    logger = current_app.logger
    camera = Camera_Settings.query.get_or_404(cam_id)
    if not camera.cam_is_active or camera.role_camera != Camera_Settings.ROLE_PRESENCE: # ROLE_PRESENCE should be defined in model
         flash(f"Camera '{camera.cam_name}' is not an active presence camera.", "warning")
         # Redirect to a general camera list or dashboard
         return redirect(url_for('admin.manage_cameras') if current_user.has_role('admin') else url_for('main.dashboard'))


    all_presence_cameras = Camera_Settings.query.filter_by(cam_is_active=True, role_camera=Camera_Settings.ROLE_PRESENCE).order_by(Camera_Settings.cam_name).all()

    # Ensure AI models are loaded (this might be redundant if your app init does it robustly)
    if face_detector is None or resnet_embedder is None:
        logger.info("AI models not loaded for presence_stream_page, attempting init.")
        with current_app.app_context(): # Ensure context for initialization
             initialize_ai_models(current_app)
        if face_detector is None or resnet_embedder is None:
            flash("AI System is not ready. Please contact administrator.", "danger")
            return redirect(url_for('main.dashboard')) # Fallback redirect

    return render_template('face/presence_stream.html',
                           camera=camera,
                           all_presence_cameras=all_presence_cameras,
                           default_selected_camera_id=cam_id)

# --- SSE Endpoint for Presence Notifications ---
@bp.route('/presence_events')

@login_required # Pastikan hanya user yang login bisa akses

def presence_events():

    def event_stream():

        global presence_message_queue

        logger = current_app.logger # Gunakan logger Flask

        logger.info("SSE: Koneksi event stream dibuka.")

        announced_messages = set() # Untuk menghindari duplikasi pesan yang sama persis dalam waktu singkat


        try:

            while True:

                try:

                    # Ambil pesan dari antrian (non-blocking)

                    message_data = presence_message_queue.get_nowait()

                    message_json = json.dumps(message_data) # Konversi dict ke string JSON


                    # Cek duplikasi sederhana

                    if message_json not in announced_messages:

                        yield f"data: {message_json}\n\n" # Format SSE

                        announced_messages.add(message_json)

                        # Bersihkan announced_messages secara periodik agar tidak terlalu besar

                        if len(announced_messages) > 50: # Batas sederhana

                            announced_messages.clear()

                        logger.info(f"SSE: Mengirim event: {message_json}")

                    else:

                        logger.debug(f"SSE: Pesan duplikat dilewati: {message_json}")

                    

                    presence_message_queue.task_done() # Tandai item selesai diproses

                except queue.Empty:

                    # Tidak ada pesan baru, kirim comment untuk menjaga koneksi tetap hidup jika perlu

                    # Atau cukup tunggu. Browser akan reconnect jika koneksi timeout.

                    yield ": keepalive\n\n" # Komentar SSE untuk keep-alive

                    time.sleep(1) # Tunggu 1 detik sebelum cek antrian lagi

                except Exception as e_sse_loop:

                    print(f"SSE: Error dalam loop event stream: {e_sse_loop}", exc_info=True)

                    time.sleep(3) # Tunggu lebih lama jika ada error

        except GeneratorExit:

            logger.info("SSE: Koneksi event stream ditutup oleh klien.")

        except Exception as e_sse_main:

            print(f"SSE: Error kritis pada event stream: {e_sse_main}", exc_info=True)

        finally:

            logger.info("SSE: Event stream loop berakhir.")


    # stream_with_context memastikan konteks aplikasi tersedia jika diperlukan

    # untuk operasi di dalam event_stream (meskipun kita coba minimalkan)

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream") 

def process_attendance_entry(data):
    personnel_id = data.get('personnel_id')
    personnel_name_log = data.get('name', f"ID:{personnel_id}")
    datetime_str = data.get('datetime')
    image_path = data.get('image_path')
    cam_id = data.get('camera_id')
    confidence_score = data.get('confidence') # Tambahkan ini
    detection_time_start = data.get('detection_time_start') # Waktu deteksi wajah pertama kali

    print(f"Processing attendance for: p_id={personnel_id}, name={personnel_name_log}, dt={datetime_str}, cam_id={cam_id}, img_path={image_path}, conf={confidence_score:.2f}") # Perbarui print

    if not all([personnel_id, datetime_str, cam_id]):
        print(f"Data input tidak lengkap untuk process_attendance_entry: personnel_id, datetime, atau cam_id kosong.")
        return 'input_error_missing_core_data'

    try:
        detected_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        detected_date_str = detected_time.strftime('%Y-%m-%d')
    except ValueError:
        print(f"Format datetime salah untuk {personnel_name_log}: {datetime_str}")
        return 'input_error_datetime_format'

    try:
        query_personnel_check = text("SELECT id, name, company_id FROM personnels WHERE id = :personnel_id LIMIT 1")
        result_personnel_check = db.session.execute(query_personnel_check, {"personnel_id": personnel_id}).fetchone()
        if not result_personnel_check:
            print(f"Personel dengan ID '{personnel_id}' tidak ditemukan di database saat proses absensi.")
            return 'personnel_id_not_found_in_db'
        personnel_name_log = result_personnel_check.name
        personnel_company_id = result_personnel_check.company_id
    except Exception as e:
        print(f"Error saat validasi personel ID '{personnel_id}': {e}", exc_info=True) # FIX: Use print
        return 'db_error_personnel_validation'
    
    try:
        query_camera_company = text("SELECT company_id FROM camera_settings WHERE id = :cam_id LIMIT 1")
        result_camera_company = db.session.execute(query_camera_company, {"cam_id": cam_id}).fetchone()
        
        if not result_camera_company:
            print(f"Kamera dengan ID '{cam_id}' tidak ditemukan.")
            return 'camera_id_not_found'
        
        camera_company_id = result_camera_company.company_id

        if personnel_company_id is None or camera_company_id is None:
            print(f"Company ID kosong untuk personel '{personnel_name_log}' (P_Comp: {personnel_company_id}) atau kamera ID '{cam_id}' (C_Comp: {camera_company_id}).")
            return 'missing_company_id_for_validation'
        
        if personnel_company_id != camera_company_id:
            print(f"Personel '{personnel_name_log}' (Perusahaan: {personnel_company_id}) tidak berasal dari perusahaan yang sama dengan kamera ID '{cam_id}' (Perusahaan: {camera_company_id}). Absensi diabaikan.")
            return 'unauthorized_entry_company_mismatch'
    except Exception as e:
        print(f"Error saat validasi company ID untuk kamera '{cam_id}': {e}", exc_info=True) # FIX: Use print
        return 'db_error_camera_company_validation'

    role_presence_value = getattr(Camera_Settings, 'ROLE_PRESENCE', 'P')

    query_settings_and_counts = text(f"""
        SELECT
            cs.attendance_time_start, cs.attendance_time_end,
            cs.leaving_time_start, cs.leaving_time_end,
            cs.cam_is_active,
            (SELECT COUNT(*) FROM personnel_entries pe
             WHERE pe.personnel_id = :p_id AND DATE(pe.timestamp) = :d_date AND pe.presence_status = 'ONTIME') AS count_ontime,
            (SELECT COUNT(*) FROM personnel_entries pe
             WHERE pe.personnel_id = :p_id AND DATE(pe.timestamp) = :d_date AND pe.presence_status = 'LATE') AS count_late,
            (SELECT COUNT(*) FROM personnel_entries pe
             WHERE pe.personnel_id = :p_id AND DATE(pe.timestamp) = :d_date AND pe.presence_status = 'LEAVE') AS count_leave
        FROM camera_settings cs
        WHERE cs.id = :c_id AND cs.role_camera = :role_cam
    """)
    
    settings_result = None
    try:
        settings_result = db.session.execute(
            query_settings_and_counts,
            {"p_id": personnel_id, "d_date": detected_date_str, "c_id": cam_id, "role_cam": role_presence_value}
        ).fetchone()
    except Exception as e:
        print(f"Error saat mengambil pengaturan kamera & count absensi untuk cam_id {cam_id}, personel {personnel_name_log}: {e}", exc_info=True) # FIX: Use print
        return 'db_error_settings_fetch'

    if not settings_result:
        print(f"Pengaturan kamera tidak ditemukan atau bukan kamera Absensi untuk ID {cam_id} (Role: {role_presence_value}). Personel: {personnel_name_log}")
        return 'invalid_camera_or_role'
    
    if not settings_result.cam_is_active:
        print(f"Kamera ID {cam_id} tidak aktif. Personel: {personnel_name_log}")
        return 'camera_inactive'

    attendance_start_str = settings_result.attendance_time_start
    attendance_end_str = settings_result.attendance_time_end
    leaving_start_str = settings_result.leaving_time_start
    leaving_end_str = settings_result.leaving_time_end

    has_ontime_today = settings_result.count_ontime > 0
    has_late_today = settings_result.count_late > 0
    has_leave_today = settings_result.count_leave > 0
    
    try:
        attendance_start = datetime.strptime(str(attendance_start_str), '%H:%M:%S').time() if attendance_start_str else None
        attendance_end = datetime.strptime(str(attendance_end_str), '%H:%M:%S').time() if attendance_end_str else None
        leaving_start = datetime.strptime(str(leaving_start_str), '%H:%M:%S').time() if leaving_start_str else None
        leaving_end = datetime.strptime(str(leaving_end_str), '%H:%M:%S').time() if leaving_end_str else None
    except ValueError as ve:
        print(f"Format waktu salah di pengaturan kamera ID {cam_id}: {ve}. Start: {attendance_start_str}, End: {attendance_end_str}")
        return 'invalid_camera_time_format'

    current_time_only = detected_time.time()
    determined_status = 'OUT_OF_TIME'

    if attendance_start and attendance_end:
        if attendance_start <= current_time_only <= attendance_end:
            determined_status = 'ONTIME'
        elif leaving_start and current_time_only > attendance_end and current_time_only < leaving_start:
            determined_status = 'LATE'
        elif current_time_only > attendance_end:
             if not leaving_start :
                determined_status = 'LATE'
        else:
            determined_status = 'OUT_OF_TIME'

    if leaving_start and leaving_end and leaving_start <= current_time_only <= leaving_end:
        determined_status = 'LEAVE'
    
    print(f"Attendance check for {personnel_name_log}: Detected time {current_time_only}, Status determined: {determined_status}")
    print(f"Window Masuk: {attendance_start}-{attendance_end}, Pulang: {leaving_start}-{leaving_end}")
    print(f"Counts Today: Ontime={has_ontime_today}, Late={has_late_today}, Leave={has_leave_today}")

    if determined_status == 'ONTIME':
        if has_ontime_today or has_late_today:
            print(f"Duplikat ONTIME/LATE untuk {personnel_name_log}. Entri ONTIME tidak dicatat.")
            return_status = 'already_attended_on_time_or_late'
        else:
            return_status = 'success'
    elif determined_status == 'LATE':
        if has_ontime_today or has_late_today:
            print(f"Duplikat ONTIME/LATE untuk {personnel_name_log}. Entri LATE tidak dicatat.")
            return_status = 'already_attended_on_time_or_late'
        else:
            return_status = 'success'
    elif determined_status == 'LEAVE':
        if has_leave_today:
            print(f"Duplikat LEAVE untuk {personnel_name_log}. Entri LEAVE tidak dicatat.")
            return_status = 'already_left_today'
        elif not (has_ontime_today or has_late_today):
            print(f"Tidak bisa LEAVE untuk {personnel_name_log} sebelum ada catatan ONTIME atau LATE hari ini.")
            return_status = 'cannot_leave_before_attendance'
        else:
            return_status = 'success'
    elif determined_status == 'OUT_OF_TIME':
        print(f"Status OUT_OF_TIME untuk {personnel_name_log} pada {detected_time}. Entri dicatat sebagai Out of Attendance Time.")
        return_status = 'out_of_time'
    elif determined_status == 'UNKNOWN':
        print(f"Status UNKNOWN untuk {personnel_name_log} pada {detected_time}. Tidak ada entri yang dibuat.")
        return_status = 'ignored_unknown_time_slot'
    else:
        return_status = 'error_unknown_status' # Default error if status not explicitly handled

    # Hanya lakukan INSERT jika statusnya 'success' atau 'out_of_time'
    if return_status in ['success', 'out_of_time', 'ignored_unknown_time_slot']:
        query_insert = text("""
            INSERT INTO personnel_entries (camera_id, personnel_id, timestamp, presence_status, image)
            VALUES (:cam_id, :p_id, :ts, :status, :img_path)
        """)
        try:
            start_db_save_time = time.time() # Waktu mulai simpan ke database
            db.session.execute(query_insert, {
                "cam_id": cam_id,
                "p_id": personnel_id,
                "ts": detected_time,
                "status": determined_status,
                "img_path": image_path
            })
            db.session.commit()
            end_db_save_time = time.time() # Waktu selesai simpan ke database
            
            # Hitung selisih waktu deteksi wajah pertama kali dan waktu simpan ke DB selesai
            # Periksa jika detection_time_start adalah None
            if detection_time_start is not None:
                time_diff_detection_to_db = (end_db_save_time - detection_time_start) * 1000 # dalam milidetik
            else:
                time_diff_detection_to_db = -1 # Atau nilai lain yang menunjukkan tidak tersedia

            print(f"Berhasil memasukkan entri {determined_status} untuk {personnel_name_log} (ID: {personnel_id}) pada {detected_time} dari kamera {cam_id}")
            print(f"Selisih waktu (deteksi pertama kali ke simpan DB selesai): {time_diff_detection_to_db:.2f} ms") # Print selisih waktu

            # Data untuk CSV
            csv_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), # Timestamp log
                personnel_name_log,
                determined_status,
                f"{time_diff_detection_to_db:.2f}", # Selisih waktu dalam ms
                f"{confidence_score*100:.2f}%", # Confidence dalam persentase
                datetime.fromtimestamp(detection_time_start).strftime('%H:%M:%S.%f') if detection_time_start is not None else "", # Waktu deteksi wajah pertama kali
                datetime.fromtimestamp(end_db_save_time).strftime('%H:%M:%S.%f'), # Waktu simpan ke DB selesai
                "Berhasil" if return_status == 'success' else "Di luar jam kerja" # Keberhasilan
            ]
            append_to_csv(csv_data) # Simpan ke CSV

            return return_status
        except Exception as e_insert:
            db.session.rollback()
            print(f"Gagal memasukkan data absensi ke database untuk {personnel_name_log}: {e_insert}", exc_info=True) # FIX: Use print
            return 'db_error_insert_failed'
    else: # Jika status bukan 'success' atau 'out_of_time', tidak ada insert DB
        print(f"Absensi untuk {personnel_name_log} tidak dicatat ke DB karena status: {return_status}")
        # Masih bisa log ke CSV jika diinginkan, namun tanpa waktu simpan DB
        # Periksa jika detection_time_start adalah None sebelum format
        det_time_str_for_csv = datetime.fromtimestamp(detection_time_start).strftime('%H:%M:%S.%f') if detection_time_start is not None else ""
        csv_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            personnel_name_log,
            determined_status,
            "", # Selisih waktu kosong
            f"{confidence_score*100:.2f}%",
            det_time_str_for_csv, # Waktu deteksi wajah pertama kali
            "", # Waktu simpan DB kosong
            "Gagal" if "error" in return_status or "cannot" in return_status else "Duplikat/Diabaikan" # Keberhasilan
        ]
        append_to_csv(csv_data) # Simpan ke CSV
        return return_status


# Fallback for routes that might still refer to old generate_face_recognition_frames_stream
# This is the one for WorkTimer, if another route needs presence, it should call the presence stream route.
# The @bp.route('/recognize_stream/<int:cam_id>') is ALREADY DEFINED above,
# and uses generate_work_timer_stream_frames.

# The /capture_video route seems to be an older, simpler recognizer.
# It will be updated to use MTCNN as well.

@bp.route('/capture_video')
@login_required
def capture_video():
    logger = current_app.logger

    def generate_frames():
        cap = get_camera_instance(0)  # Kamera default

        try:
            if not cap or not cap.isOpened():
                err_frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(err_frame, "NO CAM", (10, 30), 0, 1, (0, 0, 255), 2)
                _, jpeg_err = cv2.imencode('.jpg', err_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_err.tobytes() + b'\r\n')
                return

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                ret_jpeg, jpeg = cv2.imencode('.jpg', frame)
                if not ret_jpeg:
                    continue

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.01)
        except GeneratorExit:
            logger.info("Capture Video Test: Client disconnected, stopping stream.")
        finally:
            release_camera_instance(0)
            release_camera_instance(1)
            logger.info("Capture Video Test: Camera released, stream stopped.")

    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# @bp.route('/capture_video') # Renamed to avoid conflict, adjust if needed
# @login_required
# def capture_video():
#     logger = current_app.logger
#     personnel_id_to_name_map = {str(p.id): p.name for p in Personnels.query.all()}
#     # logger.info(f"Capture Video Test: Personnel map loaded: {len(personnel_id_to_name_map)} entries")
#     known_ids = known_embeddings_db.get("ids", [])
#     personnel_with_embeddings = []

#     for pid in known_ids:
#         name = personnel_id_to_name_map.get(pid)
#         if name:
#             personnel_with_embeddings.append(name)

#     logger.info(f"Personnel with embeddings ({len(personnel_with_embeddings)}): {personnel_with_embeddings}")

#     def generate_frames(id_to_name_map_local):
#         global face_detector, resnet_embedder, known_embeddings_db, device # Use global MTCNN
#         # Ensure preprocess_face_for_resnet_pytorch and get_embedding_resnet_pytorch are accessible

#         cap = get_camera_instance(0) # Default camera
#         if not cap or not cap.isOpened():
#             print("Capture Video Test: Cannot open camera.")
#             # Yield error frame (optional)
#             err_frame = np.zeros((480,640,3),np.uint8); cv2.putText(err_frame,"NO CAM",(10,30),0,1,(0,0,255),2)
#             _,jpeg_err = cv2.imencode('.jpg', err_frame); yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg_err.tobytes()+b'\r\n'); return

#         CONF_THRESHOLD_MTCNN = MIN_MTCNN_CONFIDENCE_STREAM # Reuse stream confidence
#         SIM_THRESHOLD_RECOG = COSINE_SIMILARITY_THRESHOLD

#         while True:
#             ret, frame_bgr_loop = cap.read()
#             if not ret or frame_bgr_loop is None:
#                 # logger.warning("Capture Video Test: Failed to read frame.")
#                 time.sleep(0.1); continue
            
#             processed_frame_disp = frame_bgr_loop.copy()

#             # Basic Frame Validation
#             if not isinstance(processed_frame_disp, np.ndarray) or processed_frame_disp.dtype != np.uint8 or \
#                processed_frame_disp.ndim != 3 or processed_frame_disp.shape[2] != 3:
#                 logger.warning("Capture Video Test: Invalid frame type/dims. Skipping AI.")
#             else:
#                 frame_h_cv, frame_w_cv = processed_frame_disp.shape[:2]
#                 if frame_h_cv < 32 or frame_w_cv < 32:
#                      logger.warning("Capture Video Test: Frame too small. Skipping AI.")
#                 else:
#                     try:
#                         frame_pil_cv = Image.fromarray(cv2.cvtColor(frame_bgr_loop, cv2.COLOR_BGR2RGB))
#                         boxes_cv, probs_cv = face_detector.detect(frame_pil_cv)

#                         if boxes_cv is not None:
#                             for i_cv, box_cv_candidate in enumerate(boxes_cv): # Rename to avoid confusion before validation
#                                 if probs_cv[i_cv] < CONF_THRESHOLD_MTCNN: 
#                                     continue
                                
#                                 # ***** ADD VALIDATION FOR box_cv_candidate ITSELF *****
#                                 if box_cv_candidate is None or not isinstance(box_cv_candidate, np.ndarray):
#                                     logger.warning(f"Capture Video Test: Invalid box_cv_candidate type or None: {type(box_cv_candidate)}. Skipping.")
#                                     continue
                                
#                                 # Assign to box_cv now that it's known to be a NumPy array
#                                 box_cv = box_cv_candidate

#                                 # Now, check for NaN/Inf, assuming box_cv is a float array as expected from MTCNN
#                                 # If it's still not float, np.isnan will fail. We might need to ensure its dtype.
#                                 # MTCNN usually returns float32 for boxes.
#                                 if box_cv.dtype != np.float32 and box_cv.dtype != np.float64:
#                                      logger.warning(f"Capture Video Test: box_cv dtype is {box_cv.dtype}, expected float. Attempting safe check or skipping.")
#                                      # For non-float types, isnan/isinf are not directly applicable.
#                                      # This situation implies an unexpected output format from MTCNN.
#                                      # We will assume if it's not float, it's likely an error state or int, so NaN/Inf check might not apply.
#                                      # However, the original error suggests the type is problematic for isnan.
#                                      # Let's try to convert to float if it's not already, to handle edge cases.
#                                      try:
#                                          box_cv_float_check = box_cv.astype(np.float32)
#                                      except ValueError: # Could happen if it contains strings or other uncastable types
#                                          logger.warning(f"Capture Video Test: Could not cast box_cv to float for NaN/Inf check. Skipping face.")
#                                          continue
#                                 else:
#                                     box_cv_float_check = box_cv # Already a float type

#                                 if np.any(np.isnan(box_cv_float_check)) or np.any(np.isinf(box_cv_float_check)) or \
#                                    box_cv_float_check[2] - box_cv_float_check[0] <= 0 or \
#                                    box_cv_float_check[3] - box_cv_float_check[1] <= 0:
#                                     logger.debug(f"Capture Video Test: Invalid values in box_cv: {box_cv_float_check}. Skipping.")
#                                     continue
                                
#                                 # If we passed all checks, original box_cv (which should be float) can be used for ROI.
#                                 # The get_face_roi_mtcnn expects float coordinates.
#                                 face_roi_cv = get_face_roi_mtcnn(frame_bgr_loop, box_cv) # Pass original box_cv
#                                 if face_roi_cv is None or face_roi_cv.size == 0: continue
                                
#                                 face_tensor_cv = preprocess_face_for_resnet_pytorch(face_roi_cv)
#                                 if face_tensor_cv is None: continue
#                                 embedding_cv = get_embedding_resnet_pytorch(face_tensor_cv)
#                                 if embedding_cv is None: continue

#                                 display_name_cv = "Unknown"
#                                 color_cv = (0,0,255)
#                                 max_sim_cv = 0.0

#                                 if known_embeddings_db["embeddings"].size > 0:
#                                     similarities_cv = cosine_similarity(embedding_cv.reshape(1, -1), known_embeddings_db["embeddings"])
#                                     best_match_idx_cv = np.argmax(similarities_cv[0])
#                                     max_sim_cv = similarities_cv[0][best_match_idx_cv]
#                                     if max_sim_cv >= SIM_THRESHOLD_RECOG:
#                                         rec_id_str_cv = known_embeddings_db["ids"][best_match_idx_cv]
#                                         display_name_cv = id_to_name_map_local.get(rec_id_str_cv, f"ID:{rec_id_str_cv}")
#                                         color_cv = (0,255,0)
                                
#                                 x1_d, y1_d, x2_d, y2_d = [int(c) for c in box_cv]
#                                 cv2.rectangle(processed_frame_disp, (x1_d,y1_d), (x2_d,y2_d), color_cv, 2)
#                                 cv2.putText(processed_frame_disp, f"{display_name_cv} ({max_sim_cv:.2f})", (x1_d, y1_d-10), 0, 0.6, color_cv, 2)
#                     except Exception as e_cv_loop:
#                         print(f"Capture Video Test: Error in AI loop: {e_cv_loop}", exc_info=True)
#                         cv2.putText(processed_frame_disp, "AI Error", (10,30),0,0.7,(0,0,255),2)


#             ret_jpeg_cv, jpeg_cv = cv2.imencode('.jpg', processed_frame_disp)
#             if not ret_jpeg_cv:
#                 logger.warning("Capture Video Test: JPEG encode failed.")
#                 continue
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg_cv.tobytes() + b'\r\n')
#             time.sleep(0.01)
        
#         release_camera_instance(0)
#         logger.info("Capture Video Test: Camera released, stream stopped.")

#     return Response(stream_with_context(generate_frames(personnel_id_to_name_map)),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

import threading
import queue

class FrameReader:
    def __init__(self, cap, max_queue_size=1):
        self.cap = cap
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.queue.full():
                self.queue.put(frame)

    def read(self):
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None

    def stop(self):
        self.running = False
        self.thread.join()


import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Add src path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
CODE_DIR = os.path.join(SCRIPT_DIR, '../Code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src.logger import setup_logger, log_execution_time
from src import config # New config module

# Constants
BATCH_SIZE = 32

def extract_features(image_dir, output_file, output_dirs=None):
    """
    Extract features from images in image_dir and save to output_file.
    """
    if output_dirs is None:
        output_dirs = config.get_output_dirs() # Default timestamped directory
        
    logger = setup_logger(log_file=os.path.join(output_dirs['LOGS'], 'execution.log'))
    logger.info(f"Starting CNN Feature Extraction (ResNet50) for {image_dir}...")
    
    # 1. Load Pre-trained Model
    try:
        # include_top=False removes the final classification layer
        # pooling='avg' returns a 1D vector (2048,)
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return False
    
    # 2. Get list of images
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return False

    files = os.listdir(image_dir)
    # Filter for our generated images
    png_files = [f for f in files if f.endswith('.png') and f.startswith('img_')]
    
    # Sort by numerical index (img_0, img_1, ...) to maintain alignment
    try:
        png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    except Exception as e:
        logger.error(f"Error sorting files: {e}")
        return False

    full_paths = [os.path.join(image_dir, f) for f in png_files]
    logger.info(f"Found {len(full_paths)} images to process.")
    
    if len(full_paths) == 0:
        logger.error("No images found to process.")
        return False

    # 3. Process in batches
    features_list = []
    fallback_indices = []  # Fix R-4: Track zero-image fallback indices
    total_samples = len(full_paths)
    total_batches = int(np.ceil(total_samples / BATCH_SIZE))
    
    logger.info(f"Processing {total_samples} images in {total_batches} batches...")

    for i in range(0, total_samples, BATCH_SIZE):
        batch_paths = full_paths[i : i + BATCH_SIZE]
        batch_images = []
        
        for j, p in enumerate(batch_paths):
            try:
                # load_img handles resizing
                img = image.load_img(p, target_size=(224, 224))
                x = image.img_to_array(img)
                x = preprocess_input(x) # ResNet-specific preprocessing
                batch_images.append(x)
            except Exception as e:
                global_idx = i + j
                logger.warning(f"Failed to load image {p} (index {global_idx}): {e}")
                # Fix N-5: Apply preprocess_input to zero image for consistency
                fallback = preprocess_input(np.zeros((224, 224, 3)))
                batch_images.append(fallback)
                fallback_indices.append(global_idx)
            
        batch_images = np.array(batch_images)
        
        # Predict
        if batch_images.shape[0] > 0:
            batch_features = base_model.predict(batch_images, verbose=0)
            features_list.append(batch_features)
        
        if (i // BATCH_SIZE) % 10 == 0:
            logger.info(f"Processed batch {i // BATCH_SIZE + 1}/{total_batches}")
            
    # Concatenate
    if features_list:
        all_features = np.concatenate(features_list, axis=0)
        logger.info(f"Feature extraction complete. Shape: {all_features.shape}")
        
        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, all_features)
        logger.info(f"Features saved to {output_file}")
        
        # Fix R-4: Log fallback indices for verification
        if fallback_indices:
            logger.warning(f"Zero-image fallback was used for {len(fallback_indices)} samples at indices: {fallback_indices}")
            logger.warning("These samples may appear as artificial outliers in Isolation Forest.")
        
        return True
    else:
        logger.error("No features extracted.")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    
    extract_features(image_dir=args.image_dir, output_file=args.output_file)

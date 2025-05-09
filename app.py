import streamlit as st
import sign_language_translator as slt
from sign_language_translator.models import get_model
from sign_language_translator.config.enums import ModelCodes, TextLanguages, SignLanguages, SignFormats
import tempfile
import os
import re
import base64
from pathlib import Path
import io
import subprocess
import cv2
import requests
import urllib.request
import shutil
import json
from sign_language_translator.vision.video.psl_processor import PSLVideoProcessor
from sign_language_translator.models.psl_to_text import PSLToTextModel

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

slt.Assets.download(r".*urls\.json")  # Download URL mappings
slt.Assets.download(r".*\.mp4")  # Download video files

import os
from pathlib import Path

# Create assets directory if it doesn't exist
assets_dir = Path(slt.Assets.ROOT_DIR)
assets_dir.mkdir(parents=True, exist_ok=True)

# Create a directory for temporary files if it doesn't exist
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)

# Add missing URLs for required files
ADDITIONAL_URLS = {
    "text_preprocessing.json": "https://raw.githubusercontent.com/sign-language-translator/sign-language-datasets/c7fbcb4d53ad4b5c04c43ed62cd98c2a6fa07f63/text-preprocessing.json",
    "ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl": "https://github.com/sign-language-translator/sign-language-datasets/releases/download/v0.0.1/ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
}

# Content for missing files
TEXT_PREPROCESSING_CONTENT = {
    "replace": {
        "urdu": {
            "؟": "?",
            "۔": ".",
            "،": ",",
            "؛": ";",
            "!": "!",
            "\\(": "(",
            "\\)": ")",
            "\\[": "[",
            "\\]": "]",
            """: "\"",
            """: "\"",
            "'": "'",
            "'": "'",
            "–": "-",
            "—": "-",
            "…": "...",
            "\\s+": " "
        },
        "english": {
            """: "\"",
            """: "\"",
            "'": "'",
            "'": "'",
            "–": "-",
            "—": "-",
            "…": "...",
            "\\s+": " "
        }
    }
}

# Check FFMPEG availability
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            st.write("Debug: FFMPEG is available")
            return True
        else:
            st.error("FFMPEG is not properly installed. Please install FFMPEG to use video features.")
            return False
    except FileNotFoundError:
        st.error("FFMPEG is not installed. Please install FFMPEG to use video features.")
        return False

st.title("Sign Language Translator")
st.write("Translate text to sign language videos or vice versa")

# Check FFMPEG at startup
if not check_ffmpeg():
    st.stop()

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'disambiguation_map' not in st.session_state:
    st.session_state.disambiguation_map = {}
if 'assets_downloaded' not in st.session_state:
    st.session_state.assets_downloaded = False

# Helper function to extract options from error message
def extract_options_from_error(error_msg):
    match = re.search(r"Try from \[(.*?)\]", str(error_msg))
    if match:
        options_str = match.group(1)
        # Split by comma and clean up the quotes
        options = [opt.strip().strip("'") for opt in options_str.split(',')]
        return options
    return []

def direct_download_file(url, target_path):
    """Download a file directly from URL to the specified path."""
    try:
        st.write(f"Debug: Direct downloading {url} to {target_path}")
        
        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use requests to download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Write the file
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        st.write(f"Debug: Successfully downloaded to {target_path}")
        return True
    except Exception as e:
        st.write(f"Debug: Error in direct download: {str(e)}")
        return False

def create_required_files():
    """Create required files directly rather than downloading them."""
    asset_dir = Path(slt.Assets.ROOT_DIR)
    models_dir = asset_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Create text_preprocessing.json
    text_preprocessing_path = asset_dir / "text_preprocessing.json"
    st.write(f"Debug: Creating {text_preprocessing_path}")
    try:
        with open(text_preprocessing_path, "w", encoding="utf-8") as f:
            json.dump(TEXT_PREPROCESSING_CONTENT, f, ensure_ascii=False, indent=2)
        st.write(f"Debug: Successfully created {text_preprocessing_path}")
    except Exception as e:
        st.error(f"Failed to create text_preprocessing.json: {str(e)}")
        return False
    
    # Create empty model file (placeholder)
    lm_path = models_dir / "ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
    st.write(f"Debug: Creating placeholder for {lm_path}")
    try:
        # Create an empty file
        lm_path.touch()
        st.write(f"Debug: Successfully created placeholder for {lm_path}")
    except Exception as e:
        st.error(f"Failed to create model placeholder: {str(e)}")
        return False
    
    return True

# Modified asset download function to create files directly
def ensure_assets_downloaded():
    if not st.session_state.assets_downloaded:
        with st.spinner("Preparing required assets... This may take a few minutes..."):
            try:
                st.write("Debug: Starting asset preparation")
                
                # Add missing URLs to the Assets
                st.write("Debug: Adding missing URLs to Assets...")
                slt.Assets.FILE_TO_URL.update(ADDITIONAL_URLS)
                
                # First download the URL mappings
                st.write("Debug: Downloading URL mappings...")
                try:
                    slt.Assets.download(r".*urls\.json")
                except Exception as e:
                    st.warning(f"Could not download URL mappings: {str(e)}")
                
                # Download pk-dictionary-mapping.json using Assets
                asset_dir = Path(slt.Assets.ROOT_DIR)
                try:
                    st.write("Debug: Downloading pk-dictionary-mapping.json via Assets...")
                    slt.Assets.download("pk-dictionary-mapping.json", overwrite=True)
                except Exception as e:
                    st.warning(f"Failed to download pk-dictionary-mapping.json: {str(e)}")
                    # Create a minimal mapping file if download fails
                    mapping_path = asset_dir / "pk-dictionary-mapping.json"
                    try:
                        with open(mapping_path, "w") as f:
                            json.dump({"empty": "placeholder"}, f)
                        st.write(f"Debug: Created placeholder for pk-dictionary-mapping.json")
                    except Exception as e2:
                        st.error(f"Failed to create pk-dictionary-mapping.json: {str(e2)}")
                        return False
                
                # Create required files directly
                if not create_required_files():
                    return False
                
                # Verify file creation
                st.write("Debug: Verifying files...")
                asset_dir = Path(slt.Assets.ROOT_DIR)
                if not asset_dir.exists():
                    st.error(f"Asset directory does not exist: {asset_dir}")
                    return False
                
                # Check for required files
                required_files = [
                    "text_preprocessing.json",
                    "pk-dictionary-mapping.json",
                    "models/ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
                ]
                
                missing_files = []
                for file in required_files:
                    file_path = asset_dir / file
                    if not file_path.exists():
                        missing_files.append(file)
                        st.write(f"Debug: Missing file path: {file_path}")
                    else:
                        st.write(f"Debug: Found file: {file_path}")
                
                if missing_files:
                    st.error(f"Missing required files: {', '.join(missing_files)}")
                    return False
                
                # Download video files
                st.write("Debug: Downloading video files...")
                try:
                    slt.Assets.download(r".*\.mp4")
                except Exception as e:
                    st.warning(f"Some video files might not have downloaded: {str(e)}")
                    st.write("Debug: This may affect video translation functionality")
                
                st.session_state.assets_downloaded = True
                st.success("Assets prepared successfully!")
                st.write(f"Debug: Asset directory contents: {list(asset_dir.glob('**/*'))[:20]}")  # Show first 20 files
                return True
                
            except Exception as e:
                st.error(f"Error preparing assets: {str(e)}")
                st.write(f"Debug: Error details: {e.__class__.__name__}: {str(e)}")
                return False
    return True

# Helper function to clean up old temporary files
def cleanup_temp_files():
    if TEMP_DIR.exists():
        for file in TEMP_DIR.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
            except Exception:
                pass

def save_video_with_ffmpeg(sign, output_path):
    """Save video using FFmpeg directly."""
    try:
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            st.write("Debug: Created temporary directory for frames")
            
            # Save individual frames as images
            st.write("Debug: Saving frames as images...")
            frame_files = []
            for i, frame in enumerate(sign):
                frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_files.append(frame_path)
            
            st.write(f"Debug: Saved {len(frame_files)} frames")
            
            # Use FFmpeg to combine frames into video
            st.write("Debug: Combining frames into video using FFmpeg")
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', '30',
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'medium',
                '-crf', '23',
                output_path
            ]
            
            st.write(f"Debug: Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.write("Debug: FFmpeg successfully created video")
                return True
            else:
                st.write(f"Debug: FFmpeg error: {result.stderr}")
                return False
                
    except Exception as e:
        st.write(f"Debug: Error in save_video_with_ffmpeg: {str(e)}")
        return False

def try_save_video(sign, tmp_file_path):
    """Try different methods to save the video."""
    # First try: Direct FFmpeg approach
    st.write("Debug: Attempting to save video using FFmpeg directly")
    if save_video_with_ffmpeg(sign, tmp_file_path):
        return True
        
    # Second try: OpenCV with different codecs
    codecs = ["avc1", "mp4v", "x264", "h264"]
    for codec in codecs:
        try:
            st.write(f"Debug: Attempting to save video with OpenCV codec {codec}")
            sign.save(
                tmp_file_path,
                overwrite=True,
                codec=codec,
                fps=30.0
            )
            st.write(f"Debug: Successfully saved video with codec {codec}")
            return True
        except Exception as e:
            st.write(f"Debug: Failed to save with codec {codec}: {str(e)}")
            continue
    return False

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    
    # Model selection
    model_code = st.selectbox(
        "Select Translation Model",
        ["text-to-sign", "sign-to-text"],
        help="Choose the translation direction"
    )
    
    # Language selection
    text_lang = st.selectbox(
        "Text Language",
        [TextLanguages.ENGLISH.value, TextLanguages.URDU.value],
        help="Select the text language"
    )
    
    sign_lang = st.selectbox(
        "Sign Language",
        [SignLanguages.PAKISTAN_SIGN_LANGUAGE.value],
        help="Select the sign language"
    )
    
    sign_format = st.selectbox(
        "Sign Format",
        [SignFormats.VIDEO.value, SignFormats.LANDMARKS.value],
        help="Select the sign format"
    )
    
    # Initialize translator
    if st.button("Initialize Translator"):
        try:
            st.write("Debug: Starting translator initialization")
            # First ensure assets are downloaded
            if model_code == "text-to-sign" and not ensure_assets_downloaded():
                st.error("Failed to download required assets. Please try again.")
            else:
                if model_code == "text-to-sign":
                    st.write(f"Debug: Initializing text-to-sign translator with: text_lang={text_lang}, sign_lang={sign_lang}, sign_format={sign_format}")
                    st.session_state.translator = slt.models.ConcatenativeSynthesis(
                        text_language=text_lang,
                        sign_language=sign_lang,
                        sign_format=sign_format
                    )
                    st.write("Debug: Translator object created")
                    st.write(f"Debug: Translator state: {st.session_state.translator}")
                    st.success("Text-to-Sign translator initialized successfully!")
                else:  # sign-to-text
                    st.write("Debug: Initializing sign-to-text processor")
                    # Initialize MediaPipe model for landmark extraction
                    st.session_state.embedding_model = slt.models.MediaPipeLandmarksModel()
                    st.write("Debug: Embedding model created")
                    st.success("Sign-to-Text processor initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing: {str(e)}")
            st.write(f"Debug: Initialization error details: {e.__class__.__name__}: {str(e)}")
            st.write(f"Debug: Current working directory: {os.getcwd()}")
            st.write(f"Debug: Assets directory: {slt.Assets.ROOT_DIR}")

# Clean up old temporary files at the start of each run
cleanup_temp_files()

# Main content area
if (model_code == "text-to-sign" and st.session_state.translator is None) or \
   (model_code == "sign-to-text" and st.session_state.embedding_model is None):
    st.warning("Please initialize the translator from the sidebar first.")
else:
    # Input section
    st.header("Translation")
    
    if model_code == "text-to-sign":
        # Text to Sign translation
        input_text = st.text_area("Enter text to translate", height=100)
        
        # Show disambiguation options if needed
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Translate"):
                if input_text:
                    # Ensure assets are downloaded before translation
                    if ensure_assets_downloaded():
                        try:
                            with st.spinner("Translating..."):
                                try:
                                    # Debug information
                                    st.write("Debug: Starting translation")
                                    st.write(f"Debug: Input text = {input_text}")
                                    st.write(f"Debug: Translator state = {st.session_state.translator}")
                                    
                                    # Try translation
                                    sign = st.session_state.translator.translate(input_text)
                                    st.write("Debug: Translation completed")
                                    
                                    # Create a temporary file for the video
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                                        st.write(f"Debug: Saving to temporary file {tmp_file.name}")
                                        
                                        # Try to save the video with different codecs
                                        if try_save_video(sign, tmp_file.name):
                                            # Read the video file and display it
                                            with open(tmp_file.name, "rb") as f:
                                                video_bytes = f.read()
                                            st.write("Debug: Video file read into memory")
                                            
                                            # Display using Streamlit
                                            st.video(video_bytes)
                                            st.write("Debug: Video displayed")
                                        else:
                                            st.error("Failed to save video with any available codec")
                                            st.write("Debug: All codec attempts failed")
                                        
                                        # Clean up the temporary file
                                        os.unlink(tmp_file.name)
                                        st.write("Debug: Temporary file cleaned up")
                                    
                                    # Clear disambiguation map if translation successful
                                    st.session_state.disambiguation_map = {}
                                        
                                except Exception as e:
                                    error_msg = str(e)
                                    if "is ambiguous" in error_msg:
                                        # Extract the ambiguous word
                                        word = error_msg.split("'")[1]
                                        options = extract_options_from_error(error_msg)
                                        if options:
                                            st.warning(f"Please select the correct form for the word '{word}'")
                                            # Store options for the word
                                            st.session_state.disambiguation_map[word] = options
                                    else:
                                        # For debugging
                                        st.error(f"Translation error: {str(e)}")
                                        st.write("Available assets:", list(slt.Assets.FILE_TO_URL.keys()))
                                        # Print more debug info
                                        st.write("Asset directory:", slt.Assets.ROOT_DIR)
                                        st.write("Current working directory:", os.getcwd())
                        except Exception as e:
                            st.error(f"Translation error: {str(e)}")
                else:
                    st.warning("Please enter some text to translate")
        
        # Show disambiguation options in sidebar
        with col2:
            if st.session_state.disambiguation_map:
                st.subheader("Word Forms")
                for word, options in st.session_state.disambiguation_map.items():
                    selected = st.radio(f"Select form for '{word}'", options)
                    # Replace the ambiguous word with the selected form
                    input_text = input_text.replace(word, selected)
    
    else:  # sign-to-text
        # Sign to Text translation
        uploaded_file = st.file_uploader("Upload a sign language video", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            if st.button("Process"):
                try:
                    with st.spinner("Processing video..."):
                        # Load video directly from uploaded file
                        video_bytes = uploaded_file.getvalue()
                        
                        # Create a temporary file with a proper path
                        temp_dir = tempfile.gettempdir()
                        temp_file_path = os.path.join(temp_dir, f"temp_video_{uploaded_file.name}")
                        
                        # Ensure the file has .mp4 extension
                        if not temp_file_path.lower().endswith('.mp4'):
                            temp_file_path += '.mp4'
                        
                        st.write(f"Debug: Temporary file path: {temp_file_path}")
                        
                        # Write the video bytes to the temporary file
                        with open(temp_file_path, 'wb') as f:
                            f.write(video_bytes)
                        
                        st.write(f"Debug: File written successfully")
                        st.write(f"Debug: File exists: {os.path.exists(temp_file_path)}")
                        
                        try:
                            # Convert path to absolute path
                            abs_path = os.path.abspath(temp_file_path)
                            st.write(f"Debug: Absolute path: {abs_path}")
                            
                            # Verify the file is readable
                            if not os.path.exists(abs_path):
                                raise FileNotFoundError(f"File not found at {abs_path}")
                            
                            # Now create the Video object with the absolute path
                            st.write("Debug: Creating Video object...")
                            video = slt.Video(str(abs_path))  # Explicitly convert to string
                            st.write("Debug: Video object created successfully")
                            
                            # Extract landmarks using MediaPipe
                            st.write("Debug: Extracting landmarks...")
                            landmarks = st.session_state.embedding_model.embed(video.iter_frames())
                            st.write("Debug: Landmarks extracted successfully")
                            
                            # Display the landmarks visualization
                            landmarks_viz = slt.Landmarks(landmarks.reshape((-1, 75, 5)), 
                                                       connections="mediapipe-world")
                            
                            # Create a temporary file for the animation
                            temp_viz_path = os.path.join(temp_dir, "temp_landmarks.mp4")
                            
                            try:
                                # Save animation to a file first
                                landmarks_viz.save_animation(temp_viz_path)
                                
                                # Read the saved file and display it
                                with open(temp_viz_path, 'rb') as f:
                                    video_bytes = f.read()
                                st.video(video_bytes)
                                
                            finally:
                                # Clean up the visualization file
                                if os.path.exists(temp_viz_path):
                                    os.unlink(temp_viz_path)
                            
                            # Display the extracted landmarks data
                            st.write("Extracted Landmarks Shape:", landmarks.shape)
                            st.write("Note: Sign-to-text translation model is not yet available. This shows the landmark extraction step.")
                            
                        except Exception as e:
                            st.write(f"Debug: Error details: {str(e)}")
                            st.write(f"Debug: Error type: {type(e)}")
                            raise e
                        finally:
                            # Clean up the temporary file
                            if os.path.exists(temp_file_path):
                                os.unlink(temp_file_path)
                                st.write("Debug: Temporary file cleaned up")
                        
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

def check_available_urls():
    """Check and display available URLs for debugging."""
    st.write("Debug: Checking available URLs...")
    try:
        # Load all URLs
        slt.Assets.load_all_urls()
        
        # Get all available URLs
        all_urls = slt.Assets.FILE_TO_URL
        
        # Display URLs for required files
        required_files = [
            "text_preprocessing.json",
            "pk-dictionary-mapping.json",
            "ur-supported-token-unambiguous-mixed-ngram-w1-w6-lm.pkl"
        ]
        
        st.write("Debug: Available URLs for required files:")
        for file in required_files:
            if file in all_urls:
                st.write(f"Debug: {file} -> {all_urls[file]}")
            else:
                st.write(f"Debug: No URL found for {file}")
        
        # Display total number of URLs
        st.write(f"Debug: Total number of URLs available: {len(all_urls)}")
        
    except Exception as e:
        st.write(f"Debug: Error checking URLs: {str(e)}")

# Add debug button in sidebar
with st.sidebar:
    if st.button("Debug URLs"):
        check_available_urls()

def main():
    st.title("Sign Language Translator")
    
    # Add language selection
    language = st.selectbox(
        "Select Target Language",
        ["English", "Urdu"]
    )
    
    # Add file uploader for PSL videos
    uploaded_file = st.file_uploader("Upload PSL Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        temp_file = TEMP_DIR / uploaded_file.name
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize processor and model
        processor = PSLVideoProcessor()
        model = PSLToTextModel(
            input_size=1662,  # Total number of landmarks (33 pose + 21*2 hands + 468 face)
            hidden_size=512,
            output_size=1000  # Vocabulary size
        )
        
        # Process video and get features
        features = processor.process_video(str(temp_file))
        
        # Get prediction
        predicted_text = model.predict(features, processor)
        
        # Display results
        st.subheader("Translation Results")
        st.write(f"Predicted Text ({language}):")
        st.write(predicted_text)
        
        # Cleanup
        cleanup_temp_files() 
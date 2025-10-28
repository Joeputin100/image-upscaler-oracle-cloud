import argparse
import base64
import os
import io
import sys
from google.cloud import aiplatform
from PIL import Image

# --- USER CONFIGURATION ---
# !! MUST be set by the user !!
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"  # e.g., "us-central1"
# --- END CONFIGURATION ---

# The model name for upscaling on Vertex AI
MODEL_NAME = "imagegeneration@006"

def upscale_image(source_path: str, factor: str):
    """
    Upscales an image using the Vertex AI Imagen API.
    """
    
    # 1. Initialize Vertex AI Client
    # This automatically uses the GOOGLE_APPLICATION_CREDENTIALS env var
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        print(f"Error initializing Vertex AI. Did you set GOOGLE_APPLICATION_CREDENTIALS?", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Get the Imagen model for upscaling
    model = aiplatform.Model(
        f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}"
    )

    # 3. Read and encode the source image
    try:
        with open(source_path, "rb") as f:
            image_bytes = f.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading source file: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Prepare the API request
    instance = {
        "image": {"bytesBase64Encoded": encoded_image}
    }
    parameters = {
        "mode": "upscale",
        "upscaleConfig": {"upscaleFactor": factor}
    }

    # 5. Make the API call
    print(f"Sending request to Vertex AI to upscale {os.path.basename(source_path)} {factor}...", file=sys.stderr)
    try:
        response = model.predict(instances=[instance], parameters=parameters)
        
        # 6. Decode the response
        if 'bytesBase64Encoded' not in response.predictions[0]:
            print(f"Error: API response did not contain an image. {response}", file=sys.stderr)
            sys.exit(1)
            
        upscaled_base64 = response.predictions[0]['bytesBase64Encoded']
        upscaled_bytes = base64.b64decode(upscaled_base64)
        
    except Exception as e:
        print(f"Error during API call: {e}", file=sys.stderr)
        print("Please ensure Vertex AI API is enabled and your account has permissions.", file=sys.stderr)
        sys.exit(1)

    # 7. Determine output path
    source_dir, source_filename = os.path.split(source_path)
    filename_base, ext = os.path.splitext(source_filename)
    output_filename = f"{filename_base}_upscaled_{factor}{ext}"
    
    # Use source_dir if it exists, otherwise use current directory
    target_dir = source_dir if source_dir else os.getcwd()
    output_path = os.path.join(target_dir, output_filename)
    
    try:
        # Try to write to target directory
        if not os.access(target_dir, os.W_OK):
            raise PermissionError
        
        with open(output_path, "wb") as f:
            f.write(upscaled_bytes)
            
    except (PermissionError, OSError):
        # Fallback to current working directory
        print(f"Warning: Target directory '{target_dir}' not writable.", file=sys.stderr)
        output_path = os.path.join(os.getcwd(), output_filename)
        print(f"Saving to current working directory: {output_path}", file=sys.stderr)
        try:
            with open(output_path, "wb") as f:
                f.write(upscaled_bytes)
        except Exception as e:
            print(f"Error: Could not write to current working directory either: {e}", file=sys.stderr)
            sys.exit(1)

    # 8. Get stats and print to STDOUT
    file_size = os.path.getsize(output_path)
    
    # Get dimensions from bytes to avoid re-reading from disk
    img = Image.open(io.BytesIO(upscaled_bytes))
    width, height = img.size

    # Print final output to STDOUT as requested
    print(f"Source Image:     {os.path.abspath(source_path)}")
    print(f"Upscaled Image:   {os.path.abspath(output_path)}")
    print(f"New Filesize:     {file_size} bytes")
    print(f"New Dimensions:   {width}x{height}")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upscale an image using Google Vertex AI (Imagen).",
        epilog="Example: python upscale_image.py ./my_image.png x4",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source_image",
        type=str,
        help="The path to the source image file (e.g., 'my_photo.png')."
    )
    parser.add_argument(
        "upscale_factor",
        type=str,
        choices=['x2', 'x4'],
        help="The factor to upscale the image (must be 'x2' or 'x4')."
    )
    return parser.parse_args()

if __name__ == "__main__":
    if PROJECT_ID == "your-gcp-project-id":
        print("Error: Please edit the PROJECT_ID and LOCATION constants at the top of the script.", file=sys.stderr)
        sys.exit(1)
        
    args = parse_args()
    upscale_image(args.source_image, args.upscale_factor)

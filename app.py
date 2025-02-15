import os
import base64
from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import gc

app = Flask(__name__)

# Constants
MAX_DIMENSION = 1500
MIN_DIMENSION = 256
STEP_SIZE = 8


# Initialize the model globally
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Enable attention slicing for memory efficiency
    pipe.enable_attention_slicing()

    return pipe


try:
    pipe = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None


def validate_dimensions(height, width):
    """
    Validate and adjust image dimensions.

    Args:
        height (int): Desired height
        width (int): Desired width

    Returns:
        tuple: Validated (height, width)

    Raises:
        ValueError: If dimensions are invalid
    """
    # Ensure dimensions are within bounds
    if height < MIN_DIMENSION or width < MIN_DIMENSION:
        raise ValueError(f"Dimensions must be at least {MIN_DIMENSION}x{MIN_DIMENSION} pixels")

    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        raise ValueError(f"Dimensions must not exceed {MAX_DIMENSION}x{MAX_DIMENSION} pixels")

    # Round to nearest multiple of STEP_SIZE
    height = (height // STEP_SIZE) * STEP_SIZE
    width = (width // STEP_SIZE) * STEP_SIZE

    return height, width


def generate_image(prompt, height=1024, width=1024):
    """
    Generate an image using Stable Diffusion.

    Args:
        prompt (str): The text prompt for image generation
        height (int): Height of the generated image
        width (int): Width of the generated image

    Returns:
        PIL.Image: The generated image

    Raises:
        Exception: If model isn't loaded or generation fails
    """
    if pipe is None:
        raise Exception("Model not loaded properly")

    # Validate dimensions
    height, width = validate_dimensions(height, width)

    # Enhance prompt for better results
    enhanced_prompt = f"{prompt}, highly detailed, professional quality, 4k"

    try:
        # Generate the image
        with torch.no_grad():
            image = pipe(
                enhanced_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=height,
                width=width
            ).images[0]

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return image

    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_image_route():
    try:
        # Get parameters from request
        prompt = request.form.get('prompt')
        if not prompt:
            raise ValueError("Prompt is required")

        # Get dimensions with defaults
        try:
            height = int(request.form.get('height', 1024))
            width = int(request.form.get('width', 1024))
        except ValueError:
            raise ValueError("Invalid dimensions provided")

        # Generate image
        img = generate_image(prompt, height, width)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str,
            'height': height,
            'width': width
        })

    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    app.run(debug=True)
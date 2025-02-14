import os
import base64
from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

app = Flask(__name__)


# Initialize the model globally
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"  # You can also try other models
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


try:
    pipe = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None


def validate_dimensions(height, width):
    """
    Ensure height and width are divisible by 8.
    If not, adjust them to the nearest valid values.

    Args:
        height (int): Desired height.
        width (int): Desired width.

    Returns:
        tuple: Valid (height, width) divisible by 8.
    """
    height = (height // 8) * 8
    width = (width // 8) * 8
    return height, width


def generate_image(prompt, style, height=1500, width=1500):
    """
    Generate an image with the specified prompt, style, and dimensions.

    Args:
        prompt (str): The text prompt for image generation.
        style (str): The artistic style to apply to the image.
        height (int): Height of the generated image (default is 1500).
        width (int): Width of the generated image (default is 1500).

    Returns:
        PIL.Image: The generated image.
    """
    if pipe is None:
        raise Exception("Model not loaded properly")

    # Validate dimensions
    height, width = validate_dimensions(height, width)

    # Combine prompt with style
    full_prompt = f"{prompt}, {style} style, highly detailed, professional"

    # Generate the image with custom dimensions
    with torch.no_grad():
        image = pipe(
            full_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=height,
            width=width
        ).images[0]

    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_image_route():
    try:
        prompt = request.form['prompt']
        style = request.form['style']

        # Optional: Allow users to specify height and width via form inputs
        height = int(request.form.get('height', 1500))  # Default to 1500 if not provided
        width = int(request.form.get('width', 1500))  # Default to 1500 if not provided

        # Validate dimensions
        height, width = validate_dimensions(height, width)

        # Generate image using Stable Diffusion
        img = generate_image(prompt, style, height, width)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str,
            'height': height,
            'width': width
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
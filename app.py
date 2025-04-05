import os
import base64
from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import gc

app = Flask(__name__)

# Constants
MIN_DIMENSION = 256  # Minimum dimension for stability
STEP_SIZE = 8  # Ensure dimensions are multiples of 8
MAX_REASONABLE_DIM = 2048  # Warning threshold, not a hard limit

# Style presets with their prompt modifiers
STYLE_PRESETS = {
    "realistic": {
        "prompt_prefix": "realistic, photorealistic, highly detailed, professional photography, 8k, ",
        "prompt_suffix": ", hyperrealistic, photographic, detailed texture, natural lighting",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, unrealistic, low quality"
    },
    "digital_art": {
        "prompt_prefix": "digital art, digital painting, trending on artstation, highly detailed, ",
        "prompt_suffix": ", vibrant colors, sharp focus, concept art, 4k, detailed",
        "negative_prompt": "photograph, photo, realistic, blurry, grainy, noisy, low quality"
    },
    "ghibli": {
        "prompt_prefix": "studio ghibli style, anime, hayao miyazaki, hand-drawn, colorful, whimsical, ",
        "prompt_suffix": ", fantasy, dreamy, soft lighting, painterly",
        "negative_prompt": "photorealistic, 3d render, photograph, realistic, grainy, noisy"
    },
    "fantasy": {
        "prompt_prefix": "fantasy art style, magical, ethereal, mystical, ",
        "prompt_suffix": ", vibrant, detailed, epic scene, dramatic lighting, fantasy illustration",
        "negative_prompt": "mundane, realistic, photograph, modern, urban"
    },
    "cyberpunk": {
        "prompt_prefix": "cyberpunk style, neon lights, futuristic, dystopian, high tech, ",
        "prompt_suffix": ", dark atmosphere, rain, reflective surfaces, night scene, blade runner style",
        "negative_prompt": "natural, daytime, bright, cheerful, rural, historical"
    },
    "oil_painting": {
        "prompt_prefix": "oil painting, traditional art, textured canvas, brush strokes visible, ",
        "prompt_suffix": ", rich colors, classical painting technique, gallery quality",
        "negative_prompt": "digital art, 3d render, photograph, smooth, flat colors"
    },
    "none": {
        "prompt_prefix": "",
        "prompt_suffix": ", high quality, detailed",
        "negative_prompt": "low quality, blurry, distorted"
    }
}


# Initialize the model globally
def load_model():
    # Use SD 1.5 for better quality
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Use DPMSolverMultistepScheduler for faster inference with good quality
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing(1)
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing(1)

    return pipe


try:
    pipe = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None


def validate_dimensions(height, width):
    """Validate and adjust image dimensions."""
    # Ensure dimensions are at least the minimum
    if height < MIN_DIMENSION or width < MIN_DIMENSION:
        raise ValueError(f"Dimensions must be at least {MIN_DIMENSION}x{MIN_DIMENSION} pixels")

    # Warning for very large dimensions (but allow them)
    if height > MAX_REASONABLE_DIM or width > MAX_REASONABLE_DIM:
        print(f"Warning: Large dimensions requested ({width}x{height}). This may cause memory issues.")

    # Round to nearest multiple of STEP_SIZE
    height = (height // STEP_SIZE) * STEP_SIZE
    width = (width // STEP_SIZE) * STEP_SIZE

    return height, width


def generate_image(prompt, height=1024, width=1024, style="none"):
    """Generate an image with the specified style and dimensions."""
    if pipe is None:
        raise Exception("Model not loaded properly")

    # Validate dimensions
    height, width = validate_dimensions(height, width)

    # Get style preset (default to "none" if style not found)
    style_preset = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["none"])

    # Apply style to prompt
    styled_prompt = f"{style_preset['prompt_prefix']}{prompt}{style_preset['prompt_suffix']}"

    # Process prompt to handle quantity specifications
    contains_single = any(word in prompt.lower() for word in ["a ", "one ", "single ", "1 "])
    contains_plural = any(
        word in prompt.lower() for word in ["multiple ", "many ", "group ", "several ", "two ", "three ", "four "])

    # Check for proper names (capitalized words)
    words = prompt.split()
    has_name = any(word[0].isupper() and len(word) > 1 for word in words if len(word) > 1)

    # Add anti-duplication to negative prompt
    base_negative = style_preset['negative_prompt']

    if has_name or contains_single:
        # Emphasize singularity for named entities or when "a/one" is specified
        styled_prompt = f"single {styled_prompt}, solo portrait, only one person, isolated subject"
        negative_prompt = f"{base_negative}, multiple people, duplicated, clone, twins, double, two, copies, group, crowd"
    else:
        negative_prompt = base_negative

    # Determine optimal inference steps based on image size
    total_pixels = height * width
    if total_pixels > 1048576:  # 1024x1024
        inference_steps = 25  # Fewer steps for very large images
        guidance_scale = 7.5
    else:
        inference_steps = 30  # More steps for standard images
        guidance_scale = 8.0

    # Progressive generation for very large images
    max_size_per_pass = 1048576  # 1024x1024 pixels
    if total_pixels > max_size_per_pass and (height > 1024 or width > 1024):
        # Generate at lower resolution first, then upscale
        scale_factor = max(1, (total_pixels / max_size_per_pass) ** 0.5)
        gen_height = int(height / scale_factor)
        gen_width = int(width / scale_factor)

        # Ensure dimensions are multiples of 8
        gen_height = (gen_height // STEP_SIZE) * STEP_SIZE
        gen_width = (gen_width // STEP_SIZE) * STEP_SIZE

        progressive = True
    else:
        gen_height = height
        gen_width = width
        progressive = False

    try:
        # Generate with optimized settings
        with torch.no_grad():
            try:
                result = pipe(
                    styled_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    height=gen_height,
                    width=gen_width
                )
                image = result.images[0]

                # Upscale if needed
                if progressive:
                    print(f"Upscaling from {gen_width}x{gen_height} to {width}x{height}")
                    image = image.resize((width, height), Image.LANCZOS)

            except RuntimeError as e:
                # Handle out-of-memory errors
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    # Try with smaller dimensions and fewer steps
                    new_height = max(height // 2, MIN_DIMENSION)
                    new_width = max(width // 2, MIN_DIMENSION)
                    new_height = (new_height // STEP_SIZE) * STEP_SIZE
                    new_width = (new_width // STEP_SIZE) * STEP_SIZE

                    print(f"OOM error. Retrying with {new_width}x{new_height}")

                    result = pipe(
                        styled_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=20,  # Reduced steps
                        guidance_scale=7.5,
                        height=new_height,
                        width=new_width
                    )
                    image = result.images[0]
                    image = image.resize((width, height), Image.LANCZOS)
                else:
                    raise e

        # Clear memory
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

        # Get style preset
        style = request.form.get('style', 'none')

        # Generate image
        img = generate_image(prompt, height, width, style)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str,
            'height': height,
            'width': width,
            'style': style
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
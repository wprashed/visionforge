from flask import Flask, request, jsonify, render_template
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64
import gc
import re
import random
import json
import numpy as np

app = Flask(__name__)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
try:
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Use DPMSolverMultistepScheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2
    )

    pipe = pipe.to(device)

    # Enable memory optimizations if on CUDA
    if device == "cuda":
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()

    print(f"Stable Diffusion loaded successfully on {device}")
except Exception as e:
    print(f"Error loading Stable Diffusion pipeline: {e}")
    pipe = None

# Style Presets
STYLE_PRESETS = {
    "photorealistic": {
        "name": "Photorealistic",
        "prompt_prefix": "photorealistic, highly detailed, professional photography, 8k, ",
        "prompt_suffix": ", hyperrealistic, photographic, detailed texture, natural lighting",
        "negative_prompt": "cartoon, anime, illustration, painting, drawing, unrealistic, low quality"
    },
    "anime": {
        "name": "Anime",
        "prompt_prefix": "anime style, manga, 2D, ",
        "prompt_suffix": ", vibrant colors, clean lines, anime aesthetic",
        "negative_prompt": "photorealistic, 3d render, photograph, realistic, grainy, noisy"
    },
    "ghibli": {
        "name": "Ghibli",
        "prompt_prefix": "studio ghibli style, miyazaki, hand-drawn animation, painterly, ",
        "prompt_suffix": ", whimsical, soft lighting, detailed backgrounds, fantasy world, dreamy atmosphere",
        "negative_prompt": "photorealistic, 3d render, photograph, realistic, grainy, noisy, dark, gloomy"
    },
    "digital_art": {
        "name": "Digital Art",
        "prompt_prefix": "digital art, digital painting, trending on artstation, highly detailed, ",
        "prompt_suffix": ", vibrant colors, sharp focus, concept art, 4k, detailed",
        "negative_prompt": "photograph, photo, realistic, blurry, grainy, noisy, low quality"
    },
    "fantasy_art": {
        "name": "Fantasy Art",
        "prompt_prefix": "fantasy art style, magical, ethereal, mystical, ",
        "prompt_suffix": ", vibrant, detailed, epic scene, dramatic lighting, fantasy illustration",
        "negative_prompt": "mundane, realistic, photograph, modern, urban"
    },
    "pixel_art": {
        "name": "Pixel Art",
        "prompt_prefix": "pixel art style, 8-bit, retro game, ",
        "prompt_suffix": ", pixelated, low resolution, retro gaming aesthetic",
        "negative_prompt": "smooth, high resolution, detailed, realistic, photorealistic"
    },
    "abstract_art": {
        "name": "Abstract Art",
        "prompt_prefix": "abstract art, non-representational, ",
        "prompt_suffix": ", shapes, colors, expressive, modern art, contemporary",
        "negative_prompt": "realistic, figurative, representational, detailed"
    }
}

# Face-specific enhancements
FACE_ENHANCEMENTS = {
    "prompt_prefix": "perfect face, symmetrical features, detailed facial features, clear eyes, natural skin texture, ",
    "prompt_suffix": ", high quality portrait, professional portrait photography, studio lighting",
    "negative_prompt": "deformed face, distorted face, disfigured, mutated, extra limbs, extra fingers, poorly drawn face, bad anatomy, blurry, low quality"
}

# Sample prompts organized by categories
SAMPLE_PROMPTS = {
    "portraits": [
        "A portrait of a young woman with long blonde hair and blue eyes",
        "A close-up portrait of an elderly man with weathered skin and wise eyes",
        "A side profile of a woman with a flower crown and freckles",
        "A dramatic portrait of a man in Renaissance style clothing",
        "A portrait of a cyberpunk character with neon lights and implants",
        "A portrait of a fantasy elf with pointed ears and silver hair",
        "A portrait of a warrior woman with battle scars and fierce expression",
        "A portrait of a person with heterochromia (different colored eyes)",
        "A portrait of a person with traditional tribal face paint",
        "A portrait of a futuristic android with human-like features",
        "A portrait of a person with a galaxy reflected in their eyes",
        "A portrait of a person wearing an elaborate headdress",
        "A portrait of a person with tears of gold",
        "A portrait of a person with a crown of thorns",
        "A portrait of a person with a halo of light"
    ],
    "landscapes": [
        "A majestic mountain range at sunset with a lake reflection",
        "A dense, misty forest with sunbeams breaking through the trees",
        "A vast desert landscape with towering sand dunes",
        "A tropical beach paradise with crystal clear turquoise water",
        "A dramatic waterfall cascading down a cliff face",
        "A serene Japanese garden with cherry blossoms and a small pond",
        "A rolling countryside with fields of lavender at golden hour",
        "A snowy mountain peak with northern lights in the sky",
        "An underwater coral reef teeming with colorful fish",
        "A volcanic landscape with flowing lava and ash clouds",
        "A terraced rice field in Asia at sunrise",
        "A canyon with layered red rock formations",
        "A lush rainforest with exotic plants and a hidden waterfall",
        "A coastal cliff face with crashing waves during a storm",
        "A peaceful meadow filled with wildflowers and butterflies"
    ],
    "fantasy": [
        "A majestic dragon perched on a castle tower",
        "A magical floating island with waterfalls cascading off the edges",
        "A mystical forest with glowing plants and fairy lights",
        "A wizard's tower surrounded by arcane symbols and magical energy",
        "A phoenix rising from ashes, wings spread with flames",
        "A crystal cave with glowing gems and underground lake",
        "A portal between worlds with swirling magical energy",
        "A mythical unicorn in an enchanted forest glade",
        "An ancient tree of life with glowing roots and branches",
        "A steampunk airship flying through clouds at sunset",
        "A kraken attacking a sailing ship in stormy seas",
        "A fairy village built into giant mushrooms",
        "A griffin soaring through mountain peaks",
        "A magical library with floating books and staircases",
        "A battle between elemental creatures of fire and ice"
    ],
    "sci-fi": [
        "A futuristic cityscape with flying vehicles and neon lights",
        "A space station orbiting a ringed planet",
        "A cybernetic being half-human half-machine",
        "A robot uprising in an abandoned factory",
        "A spacecraft landing on an alien planet with strange vegetation",
        "A holographic interface with futuristic technology",
        "A post-apocalyptic cityscape with overgrown vegetation",
        "A cyborg with glowing circuits and mechanical parts",
        "An alien landscape with bizarre rock formations and multiple moons",
        "A futuristic laboratory with advanced scientific equipment",
        "A massive spaceship hangar with various spacecraft",
        "A cyberpunk street scene with rain and reflective puddles",
        "A robot repair shop with dismantled androids",
        "A space battle with laser beams and explosions",
        "A futuristic transportation hub with hyperloop trains"
    ],
    "animals": [
        "A majestic lion with a flowing mane in the savanna",
        "A colorful peacock with its tail feathers fully displayed",
        "A snow leopard prowling through a snowy mountain landscape",
        "A hummingbird hovering near a vibrant flower",
        "A pod of dolphins jumping out of crystal blue water",
        "A tiger walking through a misty bamboo forest",
        "A family of elephants crossing a river at sunset",
        "A wolf howling at the full moon on a winter night",
        "A chameleon changing colors on a tropical leaf",
        "A grizzly bear catching salmon in a rushing stream",
        "A flock of flamingos reflected in still water",
        "An octopus changing colors and texture underwater",
        "A monarch butterfly migration across a field of flowers",
        "A red fox in a snowy forest landscape",
        "A majestic eagle soaring over mountain peaks"
    ],
    "architecture": [
        "A futuristic skyscraper with unusual geometric shapes",
        "An ancient temple complex with intricate stone carvings",
        "A Gothic cathedral with detailed stained glass windows",
        "A traditional Japanese pagoda surrounded by cherry blossoms",
        "A modern minimalist house with glass walls and clean lines",
        "A medieval castle on a cliff overlooking the sea",
        "A floating city with buildings connected by bridges",
        "A steampunk city with brass gears and steam-powered machinery",
        "An underwater dome city with glass tunnels",
        "A treehouse village connected by rope bridges",
        "A desert city carved into sandstone cliffs",
        "A futuristic sustainable city with vertical gardens",
        "An ancient Roman colosseum restored to its former glory",
        "A cyberpunk megastructure with neon advertisements",
        "A space colony built into the side of an asteroid"
    ],
    "abstract": [
        "A swirling vortex of vibrant colors and geometric shapes",
        "Liquid metal forming abstract patterns and reflections",
        "Fractals expanding into infinite geometric patterns",
        "Cosmic nebula with swirling gases and newborn stars",
        "Abstract representation of human emotions using color and form",
        "Geometric shapes transforming and evolving in space",
        "Sound waves visualized as colorful flowing patterns",
        "The concept of time represented through abstract imagery",
        "Dreams and consciousness visualized as abstract patterns",
        "Mathematical equations represented as visual art",
        "The four elements (earth, air, fire, water) as abstract forms",
        "Neural networks visualized as glowing connections",
        "Quantum physics concepts represented abstractly",
        "The flow of energy through abstract light patterns",
        "Music visualized as abstract color and movement"
    ],
    "ghibli_inspired": [
        "A young girl standing in a field of flowers with a gentle breeze",
        "A magical forest with a small cottage and spirit creatures",
        "A flying castle in the clouds with steampunk elements",
        "A cat bus running through a nighttime forest scene",
        "A bathhouse for spirits with lanterns and steam",
        "A young witch flying on a broomstick over a coastal town",
        "A peaceful valley with a winding river and small villages",
        "A giant friendly forest spirit with glowing features",
        "A child and a large fluffy creature sitting under a tree",
        "A train traveling across a vast ocean with reflections",
        "A garden with oversized vegetables and tiny creatures",
        "An abandoned amusement park with magical elements",
        "A character walking through a field of tall grass at sunset",
        "A cozy kitchen with magical cooking implements and food",
        "A character sleeping on a giant leaf in the rain"
    ]
}


def contains_face_keywords(prompt):
    """Check if the prompt likely contains a request for a face or portrait"""
    face_keywords = [
        'face', 'portrait', 'person', 'woman', 'man', 'girl', 'boy', 'human',
        'people', 'selfie', 'headshot', 'profile'
    ]

    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in face_keywords)


def generate_image(prompt, height=512, width=512, style='photorealistic', face_enhancement=False):
    """
    Generates an image based on the given prompt and style using Stable Diffusion.

    Args:
        prompt (str): The prompt to guide the image generation.
        height (int): The height of the generated image.
        width (int): The width of the generated image.
        style (str): The style preset to use.
        face_enhancement (bool): Whether to apply face-specific enhancements.

    Returns:
        PIL.Image.Image: The generated image.
    """
    if pipe is None:
        raise Exception("Stable Diffusion pipeline not loaded.")

    # Get the style preset
    style_preset = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["photorealistic"])

    # Prepare the prompt
    final_prompt = f"{style_preset['prompt_prefix']}{prompt}{style_preset['prompt_suffix']}"
    negative_prompt = style_preset["negative_prompt"]

    # Apply face enhancements if requested or if the prompt contains face-related keywords
    if face_enhancement or contains_face_keywords(prompt):
        final_prompt = f"{FACE_ENHANCEMENTS['prompt_prefix']}{final_prompt}{FACE_ENHANCEMENTS['prompt_suffix']}"
        negative_prompt = f"{negative_prompt}, {FACE_ENHANCEMENTS['negative_prompt']}"

    # Set inference parameters
    inference_steps = 40 if face_enhancement or contains_face_keywords(prompt) else 30
    guidance_scale = 8.0 if face_enhancement or contains_face_keywords(prompt) else 7.5

    # Set a seed for more consistent face generation if needed
    generator = None
    if face_enhancement or contains_face_keywords(prompt):
        seed = random.randint(1, 2147483647)
        generator = torch.Generator(device=device).manual_seed(seed)

    # Ensure dimensions are multiples of 8
    height = (height // 8) * 8
    width = (width // 8) * 8

    # Generate the image
    with torch.no_grad():
        image = pipe(
            prompt=final_prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

    # Clean up CUDA memory
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return image


# Fix the inpaint_image function to properly handle base64 images
def inpaint_image(original_image_base64, mask_base64, prompt, style='photorealistic', face_enhancement=False):
    """
    Inpaints an image based on the given mask and prompt.

    Args:
        original_image_base64 (str): Base64 encoded original image
        mask_base64 (str): Base64 encoded mask image (white areas will be inpainted)
        prompt (str): The prompt to guide the inpainting
        style (str): The style preset to use
        face_enhancement (bool): Whether to apply face-specific enhancements

    Returns:
        PIL.Image.Image: The inpainted image
    """
    if pipe is None:
        raise Exception("Stable Diffusion pipeline not loaded.")

    # Decode base64 images - handle both with and without data URL prefix
    if ',' in original_image_base64:
        original_image_base64 = original_image_base64.split(',')[1]
    if ',' in mask_base64:
        mask_base64 = mask_base64.split(',')[1]

    original_image_data = base64.b64decode(original_image_base64)
    mask_data = base64.b64decode(mask_base64)

    original_image = Image.open(io.BytesIO(original_image_data))
    mask_image = Image.open(io.BytesIO(mask_data)).convert("RGB")

    # Convert mask to black and white (white areas will be inpainted)
    mask_image = mask_image.convert("L")

    # Get the style preset
    style_preset = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["photorealistic"])

    # Prepare the prompt
    final_prompt = f"{style_preset['prompt_prefix']}{prompt}{style_preset['prompt_suffix']}"
    negative_prompt = style_preset["negative_prompt"]

    # Apply face enhancements if requested or if the prompt contains face-related keywords
    if face_enhancement or contains_face_keywords(prompt):
        final_prompt = f"{FACE_ENHANCEMENTS['prompt_prefix']}{final_prompt}{FACE_ENHANCEMENTS['prompt_suffix']}"
        negative_prompt = f"{negative_prompt}, {FACE_ENHANCEMENTS['negative_prompt']}"

    # Set inference parameters
    inference_steps = 40 if face_enhancement or contains_face_keywords(prompt) else 30
    guidance_scale = 8.0 if face_enhancement or contains_face_keywords(prompt) else 7.5

    # Ensure dimensions are multiples of 8
    width, height = original_image.size
    width = (width // 8) * 8
    height = (height // 8) * 8
    original_image = original_image.resize((width, height))
    mask_image = mask_image.resize((width, height))

    # Generate the inpainted image
    with torch.no_grad():
        image = pipe(
            prompt=final_prompt,
            image=original_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    # Clean up CUDA memory
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return image


# Fix the optimize_image function to handle data URL prefixes
def optimize_image(image_base64, quality=85, format="png", width=None, height=None):
    """
    Optimizes an image by adjusting quality, format, and dimensions.

    Args:
        image_base64 (str): Base64 encoded image
        quality (int): Quality level (1-100) for JPEG compression
        format (str): Output format (png, jpeg, webp)
        width (int, optional): New width
        height (int, optional): New height

    Returns:
        str: Base64 encoded optimized image
    """
    # Handle data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    # Decode base64 image
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    # Resize if dimensions provided
    if width and height:
        width = int(width)
        height = int(height)
        image = image.resize((width, height), Image.LANCZOS)

    # Convert to RGB if saving as JPEG
    if format.lower() == "jpeg" and image.mode == "RGBA":
        image = image.convert("RGB")

    # Save with optimization
    buffered = io.BytesIO()

    if format.lower() == "png":
        image.save(buffered, format="PNG", optimize=True)
    elif format.lower() == "jpeg":
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
    elif format.lower() == "webp":
        image.save(buffered, format="WEBP", quality=quality)
    else:
        # Default to PNG
        image.save(buffered, format="PNG", optimize=True)

    # Get base64 encoded result
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sample_prompts')
def sample_prompts():
    return jsonify(SAMPLE_PROMPTS)


@app.route('/generate', methods=['POST'])
def generate_image_route():
    try:
        # Get parameters from request
        prompt = request.form.get('prompt')
        if not prompt:
            raise ValueError("Prompt is required")

        # Get dimensions with defaults
        try:
            height = int(request.form.get('height', 512))
            width = int(request.form.get('width', 512))
        except ValueError:
            raise ValueError("Invalid dimensions provided")

        # Get style preset
        style = request.form.get('style', 'photorealistic')

        # Check for face enhancement
        face_enhancement = request.form.get('face_enhancement', 'false').lower() == 'true'

        # Generate image
        img = generate_image(prompt, height, width, style, face_enhancement)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str,
            'height': height,
            'width': width,
            'style': style,
            'format': 'png'
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
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


@app.route('/inpaint', methods=['POST'])
def inpaint_image_route():
    try:
        # Get parameters from request
        original_image = request.form.get('original_image')
        mask = request.form.get('mask')
        prompt = request.form.get('prompt')
        style = request.form.get('style', 'photorealistic')
        face_enhancement = request.form.get('face_enhancement', 'false').lower() == 'true'

        if not original_image or not mask or not prompt:
            raise ValueError("Original image, mask, and prompt are required")

        # Inpaint image
        img = inpaint_image(original_image, mask, prompt, style, face_enhancement)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str,
            'format': 'png'
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
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


@app.route('/optimize', methods=['POST'])
def optimize_image_route():
    try:
        # Get parameters from request
        image = request.form.get('image')
        quality = int(request.form.get('quality', 85))
        format = request.form.get('format', 'png')

        # Get dimensions if provided
        width = None
        height = None
        if request.form.get('width') and request.form.get('height'):
            width = int(request.form.get('width'))
            height = int(request.form.get('height'))

        if not image:
            raise ValueError("Image is required")

        # Optimize image
        optimized_image = optimize_image(image, quality, format, width, height)

        return jsonify({
            'success': True,
            'image': optimized_image,
            'format': format
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


if __name__ == '__main__':
    app.run(debug=True)

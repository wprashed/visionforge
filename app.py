from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io
import base64
import gc
import random
import numpy as np
import os
import traceback
import json
import shutil
from datetime import datetime

# Create Flask app with increased max content length
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

# Set the maximum request body size for Werkzeug
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories for training data
TRAINING_DIR = "training_data"
IMAGES_DIR = os.path.join(TRAINING_DIR, "images")
METADATA_FILE = os.path.join(TRAINING_DIR, "metadata.json")

os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize metadata file if it doesn't exist
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w') as f:
        json.dump({"images": {}}, f)

# Load the Stable Diffusion pipeline
try:
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable safety checker for unrestricted generation
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

# Style Presets - Simplified to remove restrictions
STYLE_PRESETS = {
    "photorealistic": {
        "name": "Photorealistic",
        "prompt_prefix": "photorealistic, highly detailed, professional photography, 8k, ",
        "prompt_suffix": ", hyperrealistic, photographic, detailed texture, natural lighting",
        "negative_prompt": "cartoon, anime, sketch, painting, drawing, illustration, low quality, low resolution, blurry"
    },
    "anime": {
        "name": "Anime",
        "prompt_prefix": "anime style, manga, 2D, ",
        "prompt_suffix": ", vibrant colors, clean lines, anime aesthetic",
        "negative_prompt": "photorealistic, photograph, low quality, low resolution, blurry"
    },
    "ghibli": {
        "name": "Ghibli",
        "prompt_prefix": "studio ghibli style, miyazaki, hand-drawn animation, painterly, ",
        "prompt_suffix": ", whimsical, soft lighting, detailed backgrounds, fantasy world, dreamy atmosphere",
        "negative_prompt": "photorealistic, photograph, low quality, low resolution, blurry"
    },
    "digital_art": {
        "name": "Digital Art",
        "prompt_prefix": "digital art, digital painting, trending on artstation, highly detailed, ",
        "prompt_suffix": ", vibrant colors, sharp focus, concept art, 4k, detailed",
        "negative_prompt": "low quality, low resolution, blurry"
    },
    "fantasy_art": {
        "name": "Fantasy Art",
        "prompt_prefix": "fantasy art style, magical, ethereal, mystical, ",
        "prompt_suffix": ", vibrant, detailed, epic scene, dramatic lighting, fantasy illustration",
        "negative_prompt": "low quality, low resolution, blurry"
    },
    "pixel_art": {
        "name": "Pixel Art",
        "prompt_prefix": "pixel art style, 8-bit, retro game, ",
        "prompt_suffix": ", pixelated, low resolution, retro gaming aesthetic",
        "negative_prompt": "photorealistic, photograph, high resolution"
    },
    "abstract_art": {
        "name": "Abstract Art",
        "prompt_prefix": "abstract art, non-representational, ",
        "prompt_suffix": ", shapes, colors, expressive, modern art, contemporary",
        "negative_prompt": "photorealistic, photograph, low quality, blurry"
    }
}

# Face-specific enhancements - Enhanced for better face generation
FACE_ENHANCEMENTS = {
    "prompt_prefix": "detailed facial features, high quality face details, symmetrical face, realistic skin texture, ",
    "prompt_suffix": ", high quality portrait, professional portrait photography, studio lighting, 8k uhd, perfect face proportions, clear eyes, detailed iris",
    "negative_prompt": "deformed face, distorted face, disfigured, bad anatomy, extra limbs, poorly drawn face, mutation, mutated, extra fingers, malformed limbs, too many fingers, long neck, cross-eyed, mutated hands, bad hands, bad eyes, poorly drawn hands, missing limb"
}

# Full-body enhancements - Enhanced for better body generation
FULL_BODY_ENHANCEMENTS = {
    "prompt_prefix": "full body shot, full figure, standing pose, anatomically correct, proper proportions, ",
    "prompt_suffix": ", full length portrait, entire body visible, detailed clothing, perfect anatomy",
    "negative_prompt": "deformed, distorted, disfigured, bad anatomy, extra limbs, poorly drawn, mutation, mutated, extra fingers, malformed limbs, too many fingers, long neck, cross-eyed, mutated hands, bad hands, missing limb"
}

# Sample prompts organized by categories
SAMPLE_PROMPTS = {
    "portraits": [
        "A full-body portrait of a young woman wearing a floral dress, standing in a garden",
        "A full-body portrait of a man in a suit, standing confidently against a city skyline",
        "A full-body portrait of a person with tattoos, standing in a forest at sunset",
        "A full-body portrait of a futuristic soldier in armor, standing ready for battle",
        "A full-body portrait of a fantasy character with wings, standing in a magical forest",
        "A full-body portrait of a person with traditional attire, standing in a historical setting",
        "A full-body portrait of a person with glowing eyes, standing in a dark room",
        "A full-body portrait of a person with a galaxy reflected in their eyes, standing in space",
        "A full-body portrait of a person with an elaborate headdress, standing in a ceremonial hall",
        "A full-body portrait of a person with tears of gold, standing in a mystical landscape",
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
    if prompt is None:
        return False
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in face_keywords)

def contains_full_body_keywords(prompt):
    """Check if the prompt likely contains a request for a full body image"""
    full_body_keywords = [
        'full body', 'full-body', 'standing', 'full figure', 'full length', 'head to toe',
        'entire body', 'whole body', 'full shot', 'full portrait'
    ]
    if prompt is None:
        return False
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in face_keywords)

def generate_image(prompt, height=1024, width=768, style='photorealistic', face_enhancement=False):
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

    # Check if the prompt contains human/face-related terms
    has_face_terms = contains_face_keywords(prompt)
    has_full_body_terms = contains_full_body_keywords(prompt)
    
    # Prepare the prompt
    final_prompt = f"{style_preset['prompt_prefix']}{prompt}{style_preset['prompt_suffix']}"
    
    # Prepare negative prompt - avoid low quality by default
    negative_prompt = style_preset['negative_prompt']
    
    # Only apply face enhancements if explicitly requested by checkbox OR prompt contains face terms
    if (face_enhancement and has_face_terms) or (style == 'photorealistic' and has_face_terms):
        final_prompt = f"{FACE_ENHANCEMENTS['prompt_prefix']}{final_prompt}{FACE_ENHANCEMENTS['prompt_suffix']}"
        # Use face enhancement negative prompt if provided
        if FACE_ENHANCEMENTS['negative_prompt']:
            negative_prompt = FACE_ENHANCEMENTS['negative_prompt']
    
    # Only apply full body enhancements if explicitly requested in the prompt
    if has_full_body_terms:
        final_prompt = f"{FULL_BODY_ENHANCEMENTS['prompt_prefix']}{final_prompt}{FULL_BODY_ENHANCEMENTS['prompt_suffix']}"
        # Use full body enhancement negative prompt if provided
        if FULL_BODY_ENHANCEMENTS['negative_prompt']:
            negative_prompt = FULL_BODY_ENHANCEMENTS['negative_prompt']
        
        # For full body shots, use a wider aspect ratio
        if height > width * 1.5:  # If too tall and narrow
            height = int(width * 1.5)  # Use 3:2 aspect ratio for full body

    # Log the final prompt and negative prompt for debugging
    print(f"Final prompt: {final_prompt}")
    print(f"Negative prompt: {negative_prompt}")

    # Set inference parameters - increased for better quality
    inference_steps = 40
    guidance_scale = 7.5  # Slightly reduced to avoid over-filtering

    # Set a seed for more consistent generation
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
            generator=generator,
            num_images_per_prompt=1
        ).images[0]

    # Clean up CUDA memory
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return image

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

    try:
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

        # Check if the prompt contains human/face-related terms
        has_face_terms = contains_face_keywords(prompt)
        has_full_body_terms = contains_full_body_keywords(prompt)
        
        # Prepare the prompt
        final_prompt = f"{style_preset['prompt_prefix']}{prompt}{style_preset['prompt_suffix']}"
        
        # Prepare negative prompt - avoid low quality by default
        negative_prompt = style_preset['negative_prompt']
        
        # Only apply face enhancements if explicitly requested by checkbox OR prompt contains face terms
        if (face_enhancement and has_face_terms) or (style == 'photorealistic' and has_face_terms):
            final_prompt = f"{FACE_ENHANCEMENTS['prompt_prefix']}{final_prompt}{FACE_ENHANCEMENTS['prompt_suffix']}"
            # Use face enhancement negative prompt if provided
            if FACE_ENHANCEMENTS['negative_prompt']:
                negative_prompt = FACE_ENHANCEMENTS['negative_prompt']
        
        # Only apply full body enhancements if explicitly requested in the prompt
        if has_full_body_terms:
            final_prompt = f"{FULL_BODY_ENHANCEMENTS['prompt_prefix']}{final_prompt}{FULL_BODY_ENHANCEMENTS['prompt_suffix']}"
            # Use full body enhancement negative prompt if provided
            if FULL_BODY_ENHANCEMENTS['negative_prompt']:
                negative_prompt = FULL_BODY_ENHANCEMENTS['negative_prompt']

        # Log the final prompt and negative prompt for debugging
        print(f"Inpaint final prompt: {final_prompt}")
        print(f"Inpaint negative prompt: {negative_prompt}")

        # Set inference parameters - increased for better quality
        inference_steps = 40
        guidance_scale = 7.5  # Slightly reduced to avoid over-filtering

        # Set a seed for more consistent generation
        seed = random.randint(1, 2147483647)
        generator = torch.Generator(device=device).manual_seed(seed)

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
                generator=generator,
                num_images_per_prompt=1
            ).images[0]

        # Clean up CUDA memory
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return image
    except Exception as e:
        print(f"Error in inpaint_image: {e}")
        print(traceback.format_exc())
        raise

def compress_image_base64(image_base64, max_size_kb=1000):
    """
    Compresses an image to reduce its size while maintaining quality.
    Args:
        image_base64 (str): Base64 encoded image
        max_size_kb (int): Maximum size in KB
    Returns:
        str: Base64 encoded compressed image
    """
    try:
        # Handle data URL prefix if present
        prefix = ""
        if ',' in image_base64:
            prefix, image_base64 = image_base64.split(',', 1)
            prefix += ','

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Initial quality and size
        quality = 85
        max_size_bytes = max_size_kb * 1024
        
        # First, resize large images immediately
        width, height = image.size
        max_dimension = 1600  # Limit maximum dimension
        
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Try to compress the image
        while quality > 20:  # Lower minimum quality to 20
            buffered = io.BytesIO()
            
            # Convert to RGB if needed
            if image.mode == "RGBA":
                image = image.convert("RGB")
                
            # Save with compression
            image.save(buffered, format="JPEG", quality=quality, optimize=True)
            
            # Check size
            if buffered.tell() <= max_size_bytes:
                break
                
            # Reduce quality and try again
            quality -= 15  # More aggressive quality reduction
            
        # If still too large, resize the image
        if buffered.tell() > max_size_bytes:
            # Calculate new dimensions to reduce size
            width, height = image.size
            ratio = (max_size_bytes / buffered.tell()) ** 0.5  # Square root to apply to both dimensions
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize and compress again
            image = image.resize((new_width, new_height), Image.LANCZOS)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=quality, optimize=True)
        
        # Get base64 encoded result
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return prefix + img_str
    except Exception as e:
        print(f"Error in compress_image_base64: {e}")
        print(traceback.format_exc())
        # Return original if compression fails
        return image_base64

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
    try:
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
    except Exception as e:
        print(f"Error in optimize_image: {e}")
        print(traceback.format_exc())
        raise

def save_training_image(image_base64, name, category=None):
    """
    Saves an image for training purposes.
    Args:
        image_base64 (str): Base64 encoded image
        name (str): Name/label for the image
        category (str, optional): Category for the image
    Returns:
        dict: Information about the saved image
    """
    try:
        # Handle data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{name.replace(' ', '_')}_{timestamp}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Save the image
        image.save(filepath, format="PNG")
        
        # Update metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        metadata["images"][filename] = {
            "name": name,
            "category": category,
            "timestamp": timestamp,
            "path": filepath
        }
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "filename": filename,
            "name": name,
            "category": category,
            "timestamp": timestamp
        }
    except Exception as e:
        print(f"Error in save_training_image: {e}")
        print(traceback.format_exc())
        raise

def get_training_data():
    """
    Gets all training data.
    Returns:
        dict: All training data
    """
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error in get_training_data: {e}")
        print(traceback.format_exc())
        return {"images": {}}

def delete_training_image(filename):
    """
    Deletes a training image.
    Args:
        filename (str): The filename of the image to delete
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Check if file exists in metadata
        if filename not in metadata["images"]:
            return False
        
        # Get file path
        filepath = metadata["images"][filename]["path"]
        
        # Delete file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Remove from metadata
        del metadata["images"][filename]
        
        # Save updated metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error in delete_training_image: {e}")
        print(traceback.format_exc())
        return False

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
            height = int(request.form.get('height', 1024))
            width = int(request.form.get('width', 768))
        except ValueError:
            raise ValueError("Invalid dimensions provided")

        style = request.form.get('style', 'photorealistic')
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
        print(f"ValueError in generate_image_route: {ve}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        print(f"Exception in generate_image_route: {e}")
        print(traceback.format_exc())
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

        # Compress images with more aggressive settings
        original_image = compress_image_base64(original_image, max_size_kb=800)
        mask = compress_image_base64(mask, max_size_kb=500)

        # Inpaint image
        img = inpaint_image(original_image, mask, prompt, style, face_enhancement)

        # Convert to base64 with compression
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str,
            'format': 'jpeg'
        })
    except ValueError as ve:
        print(f"ValueError in inpaint_image_route: {ve}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        print(f"Exception in inpaint_image_route: {e}")
        print(traceback.format_exc())
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
        
        # Compress image if it's too large
        image = compress_image_base64(image)
        
        # Optimize image
        optimized_image = optimize_image(image, quality, format, width, height)
        
        return jsonify({
            'success': True,
            'image': optimized_image,
            'format': format
        })
    except ValueError as ve:
        print(f"ValueError in optimize_image_route: {ve}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        print(f"Exception in optimize_image_route: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/train/save', methods=['POST'])
def save_training_image_route():
    try:
        # Get parameters from request
        image = request.form.get('image')
        name = request.form.get('name')
        category = request.form.get('category')
        
        if not image or not name:
            raise ValueError("Image and name are required")
        
        # Save training image
        result = save_training_image(image, name, category)
        
        return jsonify({
            'success': True,
            'data': result
        })
    except ValueError as ve:
        print(f"ValueError in save_training_image_route: {ve}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        print(f"Exception in save_training_image_route: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/train/data', methods=['GET'])
def get_training_data_route():
    try:
        # Get training data
        data = get_training_data()
        
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        print(f"Exception in get_training_data_route: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/train/delete', methods=['POST'])
def delete_training_image_route():
    try:
        # Get parameters from request
        filename = request.form.get('filename')
        
        if not filename:
            raise ValueError("Filename is required")
        
        # Delete training image
        result = delete_training_image(filename)
        
        return jsonify({
            'success': result
        })
    except ValueError as ve:
        print(f"ValueError in delete_training_image_route: {ve}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        print(f"Exception in delete_training_image_route: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add a route to serve training images
@app.route('/training_data/images/<filename>')
def serve_training_image(filename):
    try:
        return send_from_directory(IMAGES_DIR, filename)
    except Exception as e:
        print(f"Error serving training image: {e}")
        print(traceback.format_exc())
        return "Image not found", 404

if __name__ == '__main__':
    # Increase the maximum request size for Werkzeug
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    # Set environment variable to increase max request size
    os.environ['WERKZEUG_SERVER_MAX_CONTENT_LENGTH'] = str(500 * 1024 * 1024)  # 500MB
    
    app.run(debug=True, threaded=True)
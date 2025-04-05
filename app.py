import os
import base64
from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageOps
import io
import gc
import numpy as np

app = Flask(__name__)

# Constants
MIN_DIMENSION = 256
STEP_SIZE = 8
MAX_REASONABLE_DIM = 2048

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

# Global model instances
pipe = None
inpaint_pipe = None

def load_models():
    global pipe, inpaint_pipe
    
    # Load the base generation model
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DPMSolverMultistepScheduler for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2
    )
    
    # Load the inpainting model
    inpaint_model_id = "runwayml/stable-diffusion-inpainting"
    
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        inpaint_pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2
    )
    
    # Move models to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        inpaint_pipe = inpaint_pipe.to("cuda")
        pipe.enable_attention_slicing(1)
        inpaint_pipe.enable_attention_slicing(1)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            inpaint_pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = pipe.to("cpu")
        inpaint_pipe = inpaint_pipe.to("cpu")
        pipe.enable_attention_slicing(1)
        inpaint_pipe.enable_attention_slicing(1)
    
    return pipe, inpaint_pipe

try:
    pipe, inpaint_pipe = load_models()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    pipe, inpaint_pipe = None, None

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

def generate_image(prompt, height=512, width=512, style="none"):
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
    contains_plural = any(word in prompt.lower() for word in ["multiple ", "many ", "group ", "several ", "two ", "three ", "four "])
    
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

def inpaint_image(image_base64, mask_base64, prompt, style="none"):
    """Inpaint a specific area of an image based on a mask."""
    if inpaint_pipe is None:
        raise Exception("Inpainting model not loaded properly")
    
    # Decode base64 images
    try:
        # Decode the original image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Decode the mask
        mask_data = base64.b64decode(mask_base64)
        mask = Image.open(io.BytesIO(mask_data)).convert("L")
        
        # Ensure mask is same size as image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        # Invert mask if needed (white areas are inpainted in SD)
        mask = ImageOps.invert(mask)
    except Exception as e:
        raise ValueError(f"Error processing image or mask: {str(e)}")
    
    # Get style preset
    style_preset = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["none"])
    
    # Apply style to prompt
    styled_prompt = f"{style_preset['prompt_prefix']}{prompt}{style_preset['prompt_suffix']}"
    negative_prompt = style_preset['negative_prompt']
    
    try:
        # Perform inpainting
        with torch.no_grad():
            result = inpaint_pipe(
                prompt=styled_prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=25,
                guidance_scale=7.5,
            )
            inpainted_image = result.images[0]
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return inpainted_image
    
    except Exception as e:
        raise Exception(f"Inpainting failed: {str(e)}")

def optimize_image(image_base64, format="jpeg", quality=85, resize=None):
    """Optimize an image by changing format, quality, and size."""
    try:
        # Decode the image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Resize if specified
        if resize and isinstance(resize, dict) and 'width' in resize and 'height' in resize:
            width = int(resize['width'])
            height = int(resize['height'])
            if width > 0 and height > 0:
                image = image.resize((width, height), Image.LANCZOS)
        
        # Convert to RGB if needed (for JPEG)
        if format.lower() in ['jpeg', 'jpg'] and image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save with specified format and quality
        buffered = io.BytesIO()
        
        if format.lower() in ['jpeg', 'jpg']:
            image.save(buffered, format="JPEG", quality=quality, optimize=True)
        elif format.lower() == 'png':
            image.save(buffered, format="PNG", optimize=True)
        elif format.lower() == 'webp':
            image.save(buffered, format="WEBP", quality=quality, method=6)
        else:
            # Default to JPEG
            image.save(buffered, format="JPEG", quality=quality, optimize=True)
        
        # Get file size
        file_size = buffered.tell()
        
        # Convert to base64
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'image': img_str,
            'format': format.lower(),
            'width': image.width,
            'height': image.height,
            'size_bytes': file_size,
            'size_kb': round(file_size / 1024, 2)
        }
    
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

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
            height = int(request.form.get('height', 512))
            width = int(request.form.get('width', 512))
        except ValueError:
            raise ValueError("Invalid dimensions provided")
        
        # Get style preset
        style = request.form.get('style', 'none')
        
        # Generate image
        img = generate_image(prompt, height, width, style)
        
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.route('/inpaint', methods=['POST'])
def inpaint_image_route():
    try:
        # Get parameters from request
        image_base64 = request.form.get('image')
        mask_base64 = request.form.get('mask')
        prompt = request.form.get('prompt')
        style = request.form.get('style', 'none')
        
        if not image_base64 or not mask_base64 or not prompt:
            raise ValueError("Image, mask, and prompt are required")
        
        # Remove data URL prefix if present
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',')[1]
        if mask_base64.startswith('data:image'):
            mask_base64 = mask_base64.split(',')[1]
        
        # Perform inpainting
        inpainted_img = inpaint_image(image_base64, mask_base64, prompt, style)
        
        # Convert to base64
        buffered = io.BytesIO()
        inpainted_img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': img_str,
            'height': inpainted_img.height,
            'width': inpainted_img.width,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.route('/optimize', methods=['POST'])
def optimize_image_route():
    try:
        # Get parameters from request
        image_base64 = request.form.get('image')
        format = request.form.get('format', 'jpeg')
        
        try:
            quality = int(request.form.get('quality', 85))
            quality = max(1, min(100, quality))  # Ensure quality is between 1-100
        except ValueError:
            quality = 85
        
        # Get resize parameters if provided
        resize = None
        if request.form.get('resize_width') and request.form.get('resize_height'):
            try:
                width = int(request.form.get('resize_width'))
                height = int(request.form.get('resize_height'))
                resize = {'width': width, 'height': height}
            except ValueError:
                pass
        
        if not image_base64:
            raise ValueError("Image is required")
        
        # Remove data URL prefix if present
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',')[1]
        
        # Optimize the image
        result = optimize_image(image_base64, format, quality, resize)
        
        return jsonify({
            'success': True,
            **result
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
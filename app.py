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


def generate_image(prompt, style):
    if pipe is None:
        raise Exception("Model not loaded properly")

    # Combine prompt with style
    full_prompt = f"{prompt}, {style} style, highly detailed, professional"

    # Generate the image
    with torch.no_grad():
        image = pipe(
            full_prompt,
            num_inference_steps=30,
            guidance_scale=7.5
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

        # Generate image using Stable Diffusion
        img = generate_image(prompt, style)

        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_str
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
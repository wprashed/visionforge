# Vision Forge

## Overview

**Vision Forge** is a powerful Flask-based web application designed to generate, inpaint, and optimize images using the Stable Diffusion model. This tool empowers users to create stunning visuals based on textual descriptions, modify specific areas of existing images, and enhance image quality with various styles and optimizations.

## Project Repository

You can find the project repository here: [Vision Forge on GitHub](https://github.com/wprashed/visionforge)

## Features

- **Image Generation**: Create images from textual prompts with support for multiple artistic styles.
- **Inpainting**: Modify specific regions of an image guided by a mask and prompt.
- **Image Optimization**: Adjust image quality, format, and dimensions for optimal performance.
- **Style Presets**: Choose from diverse styles including photorealistic, anime, Ghibli, digital art, and more.
- **Face Enhancements**: Special optimizations for generating high-quality portraits and faces.
- **Sample Prompts**: Access categorized sample prompts for quick testing and inspiration.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (optional but recommended for faster inference)

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/wprashed/visionforge.git
   cd visionforge
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights**

   The script automatically downloads the necessary model weights from Hugging Face. Ensure you have sufficient disk space (~7GB).

4. **Run the Application**

   ```bash
   python app.py
   ```

   The application will be available at [http://localhost:5000](http://localhost:5000).

## Usage

### Endpoints

#### 1. Home Page

- **URL**: `/`
- **Method**: `GET`
- **Description**: Renders the main index page.

#### 2. Sample Prompts

- **URL**: `/sample_prompts`
- **Method**: `GET`
- **Description**: Returns a JSON object containing categorized sample prompts.

**Example Response:**

```json
{
    "portraits": [
        "A portrait of a young woman with long blonde hair and blue eyes",
        ...
    ],
    "landscapes": [
        "A majestic mountain range at sunset with a lake reflection",
        ...
    ],
    ...
}
```

#### 3. Generate Image

- **URL**: `/generate`
- **Method**: `POST`
- **Parameters**:
  - `prompt` (required): Textual description for image generation.
  - `height` (optional, default=512): Image height.
  - `width` (optional, default=512): Image width.
  - `style` (optional, default="photorealistic"): Style preset.
  - `face_enhancement` (optional, default=false): Apply face-specific enhancements.

**Example Request:**

```bash
curl -X POST http://localhost:5000/generate \
     -F "prompt=A futuristic cityscape with flying vehicles and neon lights" \
     -F "height=512" \
     -F "width=512" \
     -F "style=sci-fi" \
     -F "face_enhancement=false"
```

**Example Response:**

```json
{
    "success": true,
    "image": "base64_encoded_image_string",
    "height": 512,
    "width": 512,
    "style": "sci-fi",
    "format": "png"
}
```

#### 4. Inpaint Image

- **URL**: `/inpaint`
- **Method**: `POST`
- **Parameters**:
  - `original_image` (required): Base64 encoded original image.
  - `mask` (required): Base64 encoded mask image (white areas will be inpainted).
  - `prompt` (required): Textual description guiding the inpainting.
  - `style` (optional, default="photorealistic"): Style preset.
  - `face_enhancement` (optional, default=false): Apply face-specific enhancements.

**Example Request:**

```bash
curl -X POST http://localhost:5000/inpaint \
     -F "original_image=base64_encoded_original_image" \
     -F "mask=base64_encoded_mask_image" \
     -F "prompt=Add a magical glowing effect to the selected area" \
     -F "style=fantasy_art" \
     -F "face_enhancement=false"
```

**Example Response:**

```json
{
    "success": true,
    "image": "base64_encoded_inpainted_image_string",
    "format": "png"
}
```

#### 5. Optimize Image

- **URL**: `/optimize`
- **Method**: `POST`
- **Parameters**:
  - `image` (required): Base64 encoded image.
  - `quality` (optional, default=85): Quality level (1-100) for JPEG compression.
  - `format` (optional, default="png"): Output format (`png`, `jpeg`, `webp`).
  - `width` (optional): New width.
  - `height` (optional): New height.

**Example Request:**

```bash
curl -X POST http://localhost:5000/optimize \
     -F "image=base64_encoded_image" \
     -F "quality=90" \
     -F "format=jpeg" \
     -F "width=800" \
     -F "height=600"
```

**Example Response:**

```json
{
    "success": true,
    "image": "base64_encoded_optimized_image_string",
    "format": "jpeg"
}
```

## Style Presets

The following style presets are available:

- **Photorealistic**
- **Anime**
- **Ghibli**
- **Digital Art**
- **Fantasy Art**
- **Pixel Art**
- **Abstract Art**

Each style comes with predefined prompt prefixes, suffixes, and negative prompts to guide the image generation process effectively.

## Face Enhancements

When generating portraits or any image that includes faces, enabling `face_enhancement` applies additional optimizations to ensure high-quality facial features, symmetrical proportions, and natural skin textures.

## Sample Prompts

The `/sample_prompts` endpoint provides a categorized list of sample prompts under various themes:

- **Portraits**
- **Landscapes**
- **Fantasy**
- **Sci-Fi**
- **Animals**
- **Architecture**
- **Abstract**
- **Ghibli Inspired**

Use these prompts as a starting point or for inspiration when creating your own.

## Optimization

Generated images can be optimized for better performance and quality:

- **Quality Adjustment**: Control the compression level for JPEGs.
- **Format Conversion**: Convert images to different formats like PNG, JPEG, or WebP.
- **Resizing**: Adjust image dimensions to fit specific requirements.

## Deployment

For production deployment, consider the following:

- **Use a Production Server**: Replace Flask's built-in server with a production-ready server like Gunicorn.
  
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```

- **Environment Variables**: Manage sensitive configurations using environment variables.
- **Containerization**: Dockerize the application for consistent deployment across environments.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Stable Diffusion**: Thanks to the researchers and developers behind the Stable Diffusion model.
- **Hugging Face**: For providing the model weights and the `diffusers` library.
- **Flask**: For the lightweight web framework.

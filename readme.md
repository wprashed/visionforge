# Stable Diffusion Image Generator

This is a Flask-based web application that generates images using the **Stable Diffusion** model. Users can provide a text prompt and select an artistic style to generate highly detailed and professional images. The application also supports custom image dimensions (height and width) as long as they are divisible by 8.

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Endpoints](#api-endpoints)
6. [Notes](#notes)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features
- Generate images based on user-provided text prompts and styles.
- Support for custom image dimensions (height and width must be divisible by 8).
- Web-based interface for easy interaction.
- GPU acceleration for faster image generation (if available).

---

## Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- PyTorch (`torch`)
- Hugging Face `diffusers` library
- Flask
- PIL (Pillow)
- CUDA (optional, for GPU acceleration)

You can install the required dependencies using the following command:

```bash
pip install torch torchvision diffusers flask pillow
```

If you plan to use GPU acceleration, ensure you have a compatible NVIDIA GPU with CUDA installed.

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stable-diffusion-generator.git
   cd stable-diffusion-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Stable Diffusion model:
   - The model (`runwayml/stable-diffusion-v1-5`) will be downloaded automatically when you run the application for the first time. Ensure you have sufficient disk space (~7GB).

---

## Usage
1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`.

3. Use the form to input:
   - **Prompt**: A description of the image you want to generate.
   - **Style**: An artistic style (e.g., "impressionist", "realistic").
   - **Height** and **Width**: Custom dimensions for the generated image (must be divisible by 8).

4. Click "Generate Image" to create and view the generated image.

---

## API Endpoints
### 1. `/`
- **Method**: `GET`
- **Description**: Renders the main HTML page with the image generation form.

### 2. `/generate`
- **Method**: `POST`
- **Description**: Generates an image based on the provided prompt, style, and dimensions.
- **Request Body**:
  ```json
  {
      "prompt": "A futuristic cityscape",
      "style": "cyberpunk",
      "height": 768,
      "width": 768
  }
  ```
- **Response**:
  - On success:
    ```json
    {
        "success": true,
        "image": "base64_encoded_image_string",
        "height": 768,
        "width": 768
    }
    ```
  - On failure:
    ```json
    {
        "success": false,
        "error": "Error message"
    }
    ```

---

## Notes
1. **Image Dimensions**: The height and width of the generated image must be divisible by 8. If invalid dimensions are provided, they will be adjusted to the nearest valid values.
2. **Performance**: Generating large images requires significant computational resources. If you encounter out-of-memory errors, reduce the resolution or use a smaller model.
3. **GPU Acceleration**: For faster generation, ensure you have a compatible GPU and CUDA installed. The application will automatically use the GPU if available.

---

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

---

## Acknowledgments
- Thanks to the creators of the [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) model for making it publicly available.
- Special thanks to the Hugging Face team for their `diffusers` library, which simplifies working with diffusion models.
# Signature-validation-system

A web-based application that compares handwritten signatures to assess their similarity using image processing techniques.

## Features
- User login interface
- Upload and compare signature images
- Image similarity analysis using SSIM
- Display comparison results through a web interface

## Technologies Used
- Python
- Flask
- OpenCV
- scikit-image
- HTML, CSS

## How to Run

1. Install Python 3.
2. Install the required libraries:

   ```bash
   pip install flask opencv-python scikit-image numpy pillow
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Open the local URL displayed in the terminal in your browser.

## Note
This project demonstrates image-based signature similarity analysis. Similarity scores alone do not establish the authenticity of a signature.

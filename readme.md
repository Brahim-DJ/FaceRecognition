# Face Recognition Streamlit App

This project implements a face recognition system using the Labeled Faces in the Wild (LFW) dataset, FaceNet for face embedding generation, and FAISS for efficient similarity search. The application is built with Streamlit for an easy-to-use web interface.

## Features

- Load and process the LFW dataset
- Generate face embeddings using FaceNet (InceptionResnetV1)
- Perform fast similarity search using FAISS
- User-friendly web interface for uploading images and finding similar faces

## Requirements

- Python 3.7+
- streamlit
- opencv-python
- numpy
- facenet-pytorch
- torch
- faiss-cpu (or faiss-gpu for CUDA support)
- Pillow

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Brahim-DJ/FaceRecognition.git
   cd FaceRecognition
   ```
2. Create a virtual python environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Linux/MacOs:
      ```
      source ./venv/bin/activate
      ```
   - On Windows:
     ```
     ./venv/Scripts/activate
     ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Download the LFW dataset and update the `LFW_DATASET_PATH` in the script.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. The first time you run the app, it will process the LFW dataset. This may take some time depending on your hardware.

4. Once the processing is complete, you can upload an image to find similar faces in the LFW dataset.

## How it Works

1. The app loads the LFW dataset and generates face embeddings using FaceNet.
2. These embeddings are stored in a FAISS index for efficient similarity search.
3. When you upload an image, the app:
   - Detects faces using MTCNN
   - Generates an embedding for the detected face using FaceNet
   - Searches for similar face embeddings in the FAISS index
   - Displays the most similar face from the LFW dataset

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: List of Python dependencies
- `README.md`: This file
- `db`: LFW generated embeddings 

import streamlit as st
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import faiss
import os
from PIL import Image

# Constants
LFW_DATASET_PATH = "./lfw-dataset" # Change this to the path of your LFW dataset
EMBEDDINGS_PATH = "./db/embeddings.npy"
LABELS_PATH = "./db/labels.npy"
INDEX_PATH = "./db/faiss_index.bin"

# Initialize FaceNet model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_lfw_dataset(dataset_path):
    images = []
    labels = []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        img_copy = img.copy()
                    images.append(img_copy)
                    labels.append(person_name)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    return images, labels

def generate_embeddings(images):
    embeddings = []
    for image in images:
        face = mtcnn(image)
        if face is not None:
            embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
            embeddings.append(embedding)
    return np.array(embeddings)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_data(embeddings, labels, index):
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)
    faiss.write_index(index, INDEX_PATH)

def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    index = faiss.read_index(INDEX_PATH)
    return embeddings, labels, index

def find_similar_face(query_embedding, index, labels, k=1):
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return [labels[i] for i in indices[0]], distances[0]

def process_image(image):
    face = mtcnn(image)
    if face is None:
        return None
    embedding = resnet(face.unsqueeze(0)).detach().cpu().numpy()[0]
    return embedding

def main():
    st.title("Face Recognition App")

    # Check if data is already processed
    if not os.path.exists(INDEX_PATH):
        st.info("Processing LFW dataset. This may take a while...")
        images, labels = load_lfw_dataset(LFW_DATASET_PATH)
        embeddings = generate_embeddings(images)
        index = create_faiss_index(embeddings)
        save_data(embeddings, labels, index)
    else:
        embeddings, labels, index = load_data()

    st.write("Upload an image to find similar faces")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Find Similar Face"):
            embedding = process_image(image)
            if embedding is not None:
                similar_labels, distances = find_similar_face(embedding, index, labels)
                st.write(f"Most similar face: {similar_labels[0]}")
                try :
                    st.image(os.path.join(LFW_DATASET_PATH, similar_labels[0], f"{similar_labels[0]}_0001.jpg"), caption="Similar Image", use_column_width=True)
                except:
                    st.write("LFW dataset not provided.")
                st.write(f"Distance: {distances[0]:.4f}")
            else:
                st.error("No face detected in the uploaded image.")

if __name__ == "__main__":
    main()
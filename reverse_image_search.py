# Read all images from the folder
# Create embeddings for each image
# Build Annoy index using embeddings
# Read Test image
# Create embeddings for test image
# Search Annoy index for nearest neighbours



import gradio as gr
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from annoy import AnnoyIndex
import torch
import os

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Configure directories
UPLOAD_FOLDER = 'static/uploads'
IMAGE_FOLDER = 'static/images'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Function to extract image embedding using CLIP
def extract_image_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.squeeze().numpy()



# Pre-build the Annoy index
dimension = 512  # Dimension of CLIP embeddings
annoy_index = AnnoyIndex(dimension, 'angular')
image_paths = []

#print(os.listdir(IMAGE_FOLDER))

for idx, filename in enumerate(os.listdir(IMAGE_FOLDER)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        embedding = extract_image_embedding(image_path)
        annoy_index.add_item(idx, embedding)
        image_paths.append(image_path)



num_trees = 100
annoy_index.build(num_trees)
annoy_index.save('image_embeddings.ann')

# Function to find similar images
def find_similar_images(image_path, num_matches=5):
    embedding = extract_image_embedding(image_path)
    #print(embedding[:5])    
    indices, distances = annoy_index.get_nns_by_vector(embedding, num_matches, include_distances=True)

    # print(indices)
    # print(distances)
    similar_images = [{"path": image_paths[idx], "distance": distances[i]} for i, idx in enumerate(indices)]
    return similar_images



# Function to display similar images in Gradio
def search_similar_images(uploaded_image):
    # Save the uploaded image
    uploaded_image_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
    uploaded_image.save(uploaded_image_path, quality=100)  # Save with maximum quality

    # Find similar images
    similar_images = find_similar_images(uploaded_image_path)

    # Prepare the list of image paths and distances for Gradio
    results = []
    for sim_img in similar_images:
        image = Image.open(sim_img['path']) #.resize((200, 200))  # Resize for better display
        results.append((image, f"Distance: {sim_img['distance']:.4f}"))
    
    return results

# Create the Gradio interface
iface = gr.Interface(
    fn=search_similar_images,
    inputs=gr.Image(type="pil", label="Upload an Image"),  # No resizing or cropping tool
    outputs=gr.Gallery(label="Similar Images"), #.style(columns=[2], object_fit="contain"),
    title="Similar Image Search Engine",
    description="Upload an image to find similar images from the dataset."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(debug=True)

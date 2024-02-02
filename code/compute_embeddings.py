from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np

data_dir = "/data"


def compute_embeddings():

    # embed the images
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images = [Image.open(f"{data_dir}/images/{img}") for img in os.listdir(f"{data_dir}/images")[0:10]]

    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    text_inputs = processor(text="a red car", return_tensors="pt", padding=True)
    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)

    np.save(f"{data_dir}/image_embeddings.npy", image_features.detach().numpy(), allow_pickle=False)

    # upload vectors to the qdrantf database
    qdrant_client = QdrantClient("qdrant")
    qdrant_client.recreate_collection(
        collection_name="images",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    image_embeddings = np.load(f"{data_dir}/image_embeddings.npy")

    qdrant_client.upload_collection(
        collection_name="images",
        vectors=image_embeddings,
        payload=[{"image_dir": img} for img in os.listdir(f"{data_dir}/images")[0:10]],
        ids=None,
        batch_size=256,
    )

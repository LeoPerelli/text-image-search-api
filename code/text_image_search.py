from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np


class TextImageSearch:
    def __init__(self) -> None:

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.qdrant_client = QdrantClient("qdrant")
        self.collection_name = "images"

    def nearest_images(self, query: str):

        text_input = self.processor(text=query, return_tensors="pt", padding=True)
        text_embedding = self.model.get_text_features(**text_input).detach().numpy()[0]

        nearest_elements = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=text_embedding,
            limit=5,
        )

        payloads = [element.payload for element in nearest_elements]
        return payloads

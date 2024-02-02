from fastapi import FastAPI
from fastapi.responses import Response, FileResponse
from text_image_search import TextImageSearch
import uvicorn
from PIL import Image
from compute_embeddings import compute_embeddings
from pathlib import Path

print("Computing embeddings...")
compute_embeddings()
print("Computed embeddings!")

searcher = TextImageSearch()
app = FastAPI()


@app.get("/api/search")
def search_image(text: str):
    dirs = searcher.nearest_images(text)
    img_path = Path(f"/data/images/{dirs[0]['image_dir']}")
    return FileResponse(img_path, media_type="image/jpg")


@app.get("/api/test")
def search_image(text: str):
    return f"Hi! your input text is {text}"


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)

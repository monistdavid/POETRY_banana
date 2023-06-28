from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor
from sentence_transformers.util import semantic_search

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    poetry_embeddings = load_dataset('monist/chinese_poetry')
    dataset_embeddings = torch.from_numpy(poetry_embeddings["train"].to_pandas().to_numpy()).to(torch.float)
    model_text = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1', device='cuda')
    processor = BlipProcessor.from_pretrained("IDEA-CCNL/Taiyi-BLIP-750M-Chinese")
    model_image = BlipForConditionalGeneration.from_pretrained("IDEA-CCNL/Taiyi-BLIP-750M-Chinese")

    context = {
        "dataset_embeddings": dataset_embeddings,
        "model_text": model_text,
        "processor": processor,
        "model_image": model_image
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    image = request.json.get("image")
    dataset_embeddings = context.get("dataset_embeddings")
    model_text = context.get("model_text")
    processor = context.get("processor")
    model_image = context.get("model_image")

    inputs = processor(image, return_tensors="pt")
    out = model_image.generate(**inputs)
    image_text = processor.decode(out[0], skip_special_tokens=True)
    output = model_text.encode(image_text)
    query_embeddings = torch.FloatTensor(output)
    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=1)

    return Response(
        json={"outputs": hits},
        status=200
    )


if __name__ == "__main__":
    app.serve()

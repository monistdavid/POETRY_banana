from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor, GenerationConfig
from PIL import Image
from sentence_transformers.util import semantic_search


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model_text
    global model_image
    global processor
    global dataset_embeddings

    poetry_embeddings = load_dataset('monist/chinese_poetry')
    dataset_embeddings = torch.from_numpy(poetry_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

    model_text = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1', device='cuda')
    processor = BlipProcessor.from_pretrained("IDEA-CCNL/Taiyi-BLIP-750M-Chinese", device='cuda')
    model_image = BlipForConditionalGeneration.from_pretrained("IDEA-CCNL/Taiyi-BLIP-750M-Chinese", device='cuda')


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model_text
    global model_image
    global processor
    global dataset_embeddings

    # Parse out your arguments
    image = [model_inputs.get('image', None)]
    if image is None:
        return {'message': "No image provided"}

    raw_image = Image.open('guan.jpg')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model_image.generate(**inputs)
    image_text = processor.decode(out[0], skip_special_tokens=True)
    output = model_text.encode(image_text)
    query_embeddings = torch.FloatTensor(output)
    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=1)

    result = [all_poetry[hits[0][i]['corpus_id']] for i in range(len(hits[0]))][0]

    # Return the results as a dictionary
    return result

# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import pipeline
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor, GenerationConfig


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    poetry_embeddings = load_dataset('monist/chinese_poetry')
    dataset_embeddings = torch.from_numpy(poetry_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

    model_text = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1', device='cuda')
    processor = BlipProcessor.from_pretrained("IDEA-CCNL/Taiyi-BLIP-750M-Chinese")
    model_image = BlipForConditionalGeneration.from_pretrained("IDEA-CCNL/Taiyi-BLIP-750M-Chinese")


if __name__ == "__main__":
    download_model()

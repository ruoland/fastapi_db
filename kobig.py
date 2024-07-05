# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="monologg/kobigbird-bert-base")


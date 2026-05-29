print("1 - démarrage")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("2 - os ok")

import chromadb
print("3 - chromadb ok")

from sentence_transformers import SentenceTransformer
print("4 - sentence_transformers ok")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("5 - modèle chargé ok")

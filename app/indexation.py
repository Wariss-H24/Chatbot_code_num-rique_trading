import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from sentence_transformers import SentenceTransformer
from chargement import charger_pdf, decouper_texte

DB_PATH = "../chroma_db"
COLLECTION_NAME = "code_numerique"


def indexer_pdf(pdf_path):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    texte = charger_pdf(pdf_path)
    chunks = decouper_texte(texte)

    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True).tolist()

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": os.path.basename(pdf_path)}] * len(chunks)
    )

    print(f"Indexation OK — {len(chunks)} chunks indexés")


if __name__ == "__main__":
    indexer_pdf(
        "../documents/Benin-Loi-2017-20-Portant-code-du-numerique-en-Republique-du-Benin.pdf"
    )

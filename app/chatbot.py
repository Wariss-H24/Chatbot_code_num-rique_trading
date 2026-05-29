import chromadb
from sentence_transformers import SentenceTransformer
from ollama import Client

DB_PATH = "../chroma_db"
COLLECTION_NAME = "code_numerique"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client_db = chromadb.PersistentClient(path=DB_PATH)

ollama = Client(host="http://10.46.3.3:11434")


def search(question):
    q_emb = model.encode(question).tolist()

    collection = client_db.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    return docs, metas


def generate(question, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
Tu es un assistant expert du Code du Numérique.

Règles:
- Répond uniquement avec le contexte
- Si tu ne sais pas, dis-le clairement

CONTEXTE:
{context}

QUESTION:
{question}

Réponse:
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    print("Chatbot prêt ! (tape 'quitter' pour arrêter)")
    while True:
        question = input("\nTa question : ")
        if question.lower() == "quitter":
            break
        chunks, _ = search(question)
        reponse = generate(question, chunks)
        print(f"\nRéponse : {reponse}")
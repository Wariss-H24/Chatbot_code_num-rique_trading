📘 Explication du projet Chatbot RAG
C'est quoi RAG ?
RAG veut dire Retrieval-Augmented Generation. C'est une technique qui permet à un modèle IA de répondre à des questions en se basant sur un document que tu lui as donné, au lieu d'inventer des réponses.

Le principe en 3 étapes :

Tu donnes un PDF → on le découpe en morceaux (chunks)
Quand tu poses une question → on cherche les morceaux les plus proches de ta question
On envoie ces morceaux + ta question au modèle IA → il génère une réponse précise
Architecture du projet
Loi_bot/
├── app/
│   ├── chargement.py   → Lire le PDF et le découper
│   ├── indexation.py   → Transformer les chunks en vecteurs et les stocker
│   ├── chatbot.py      → Chercher dans la base et générer une réponse
│   └── ui.py           → Interface web Streamlit
├── chroma_db/          → Base de données vectorielle (créée automatiquement)
└── documents/          → Dossier où sont stockés les PDFs
1. chargement.py — Lire et découper le PDF
Rôle
Ce fichier s'occupe de deux choses :

Lire le texte brut d'un fichier PDF
Découper ce texte en petits morceaux appelés chunks
Pourquoi découper ?
Un modèle IA ne peut pas lire 100 pages d'un coup. On découpe le texte en morceaux de 500 caractères pour pouvoir chercher précisément dans le document.

Les fonctions
charger_pdf(pdf_path)

Prend le chemin du PDF en paramètre
Utilise PdfReader pour lire chaque page
Concatène tout le texte de toutes les pages
Retourne un grand texte (string)
decouper_texte(texte)

Prend le grand texte en paramètre
Utilise RecursiveCharacterTextSplitter de LangChain pour découper
chunk_size=500 → chaque morceau fait max 500 caractères
chunk_overlap=100 → les morceaux se chevauchent de 100 caractères pour ne pas perdre le contexte entre deux morceaux
separators → il essaie de couper d'abord aux paragraphes, puis aux lignes, puis aux phrases
Retourne une liste de chunks (morceaux de texte)
Librairies utilisées
pypdf → lire les fichiers PDF
langchain_text_splitters → découper intelligemment le texte
2. indexation.py — Transformer et stocker les chunks
Rôle
Ce fichier prend les chunks créés par chargement.py et les transforme en vecteurs numériques (embeddings) pour les stocker dans une base de données.

C'est quoi un vecteur / embedding ?
Un embedding c'est une liste de nombres qui représente le sens d'un texte. Par exemple "chien" et "animal" auront des vecteurs proches, car ils sont sémantiquement liés. Ça permet de faire des recherches par sens et non par mot-clé exact.

La fonction indexer_pdf(pdf_path)
Étape 1 — Charger le modèle d'embedding

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
Ce modèle transforme du texte en vecteurs. Il supporte le français et l'anglais.

Étape 2 — Connexion à ChromaDB

client = chromadb.PersistentClient(path=DB_PATH)
ChromaDB est la base de données qui stocke les vecteurs sur le disque (dans chroma_db/).

Étape 3 — Supprimer l'ancienne collection et en créer une nouvelle

client.delete_collection(COLLECTION_NAME)
collection = client.create_collection(COLLECTION_NAME)
On repart de zéro à chaque indexation pour éviter les doublons.

Étape 4 — Lire et découper le PDF

texte = charger_pdf(pdf_path)
chunks = decouper_texte(texte)
On appelle les fonctions de chargement.py.

Étape 5 — Encoder en batch

embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True).tolist()
On encode tous les chunks en une seule fois par groupes de 64 (batch). C'est beaucoup plus rapide qu'encoder un par un.

Étape 6 — Stocker dans ChromaDB

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    documents=chunks,
    metadatas=[{"source": os.path.basename(pdf_path)}] * len(chunks)
)
ids → identifiant unique pour chaque chunk
embeddings → les vecteurs numériques
documents → le texte original du chunk
metadatas → le nom du fichier PDF source
Librairies utilisées
sentence-transformers → transformer du texte en vecteurs
chromadb → base de données vectorielle
3. chatbot.py — Chercher et générer une réponse
Rôle
Ce fichier contient la logique principale du chatbot :

Chercher les chunks les plus pertinents par rapport à une question
Envoyer ces chunks + la question au modèle IA pour générer une réponse
La fonction search(question)
Étape 1 — Encoder la question

q_emb = model.encode(question).tolist()
On transforme la question en vecteur, avec le même modèle que lors de l'indexation. C'est important d'utiliser le même modèle pour que les vecteurs soient comparables.

Étape 2 — Chercher dans ChromaDB

collection = client_db.get_collection(COLLECTION_NAME)
results = collection.query(query_embeddings=[q_emb], n_results=3)
ChromaDB compare le vecteur de la question avec tous les vecteurs stockés et retourne les 3 chunks les plus proches sémantiquement.

Étape 3 — Retourner les résultats

docs = results.get("documents", [[]])[0]
metas = results.get("metadatas", [[]])[0]
return docs, metas
On retourne les textes des chunks trouvés et leurs métadonnées (nom du fichier source).

La fonction generate(question, chunks)
Étape 1 — Construire le contexte

context = "\n\n".join(chunks)
On assemble les 3 chunks en un seul bloc de texte.

Étape 2 — Construire le prompt

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
Le prompt est l'instruction qu'on envoie au modèle IA. On lui donne le contexte (les chunks) et la question pour qu'il réponde précisément.

Étape 3 — Appeler Ollama

response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
return response["message"]["content"]
Ollama est un outil qui fait tourner des modèles IA en local sur un serveur. Ici on utilise le modèle Mistral qui tourne sur http://10.46.3.3:11434.

Librairies utilisées
sentence-transformers → encoder la question
chromadb → chercher les chunks pertinents
ollama → appeler le modèle IA Mistral
4. ui.py — Interface web Streamlit
Rôle
Ce fichier crée l'interface web que tu vois dans le navigateur. Il relie tous les autres fichiers ensemble pour offrir une expérience utilisateur complète.

Les sections
Configuration de la page

st.set_page_config(page_title="RAG - Code du Numérique", page_icon="📘", layout="centered")
Définit le titre de l'onglet du navigateur et l'icône.

CSS personnalisé On injecte du CSS directement dans Streamlit avec st.markdown(..., unsafe_allow_html=True). Cela permet de personnaliser les couleurs, les polices, les boutons, etc. Le style choisi : fond clair dominant (#f5f7fa) avec des accents sombres (#1a1a2e) et bleus (#4a90d9).

Upload du PDF

uploaded_file = st.file_uploader("📄 Charger le PDF", type="pdf")
Permet à l'utilisateur de charger un PDF depuis son ordinateur. Le fichier est sauvegardé dans documents/ avec son nom original.

Indexation

if st.button("🚀 Indexer le document"):
    indexer_pdf(pdf_path)
Quand l'utilisateur clique le bouton, on lance l'indexation du PDF uploadé.

Gestion de l'état avec session_state

if "last_question" not in st.session_state:
    st.session_state.last_question = ""
Streamlit recharge la page à chaque interaction. session_state permet de mémoriser des valeurs entre ces rechargements. Ici on mémorise la dernière question posée pour éviter de relancer une recherche si la question n'a pas changé.

Poser une question

question = st.text_input("💬 Pose ta question")
if question and question != st.session_state.last_question:
    docs, metas = search(question)
    answer = generate(question, docs)
Quand l'utilisateur tape une question et appuie sur Entrée :

On cherche les chunks pertinents avec search()
On génère la réponse avec generate()
On affiche la réponse et les sources dans des bulles stylisées
Librairies utilisées
streamlit → créer l'interface web
Résumé du flux complet
PDF
 │
 ▼
chargement.py → lit le PDF → découpe en chunks
 │
 ▼
indexation.py → encode les chunks en vecteurs → stocke dans ChromaDB
 │
 ▼
         [L'utilisateur pose une question]
 │
 ▼
chatbot.py → encode la question → cherche dans ChromaDB → envoie à Mistral
 │
 ▼
ui.py → affiche la réponse dans le navigateur
Les dépendances à installer
pip install pypdf langchain-text-splitters sentence-transformers chromadb ollama streamlit
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2"
Les commandes pour lancer le projet
# 1. Aller dans le dossier app
cd app

# 2. Lancer l'interface web (recommandé)
streamlit run ui.py

# 3. Ou tester le chatbot en ligne de commande
python chatbot.py
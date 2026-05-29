from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def charger_pdf(pdf_path):
    print("Lecture du PDF...")

    reader = PdfReader(pdf_path)

    texte = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texte += page_text + "\n"

    return texte


def decouper_texte(texte):
    print("Découpage en chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(texte)

    print(f"Chunks créés : {len(chunks)}")
    return chunks


if __name__ == "__main__":
    texte = charger_pdf("../documents/Benin-Loi-2017-20-Portant-code-du-numerique-en-Republique-du-Benin.pdf")
    chunks = decouper_texte(texte)

    print("\nExemple chunk :\n")
    print(chunks[0])
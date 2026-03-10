import os
from pathlib import Path
from dotenv import load_dotenv
from openai import embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector


load_dotenv()
for k in ("PGVECTOR_URL", "PGVECTOR_COLLECTION", "GOOGLE_API_KEY", "GOOGLE_API_MODEL"):
    if k not in os.environ:
        raise RuntimeError(f"Missing required environment variable: {k}")
    

current_dir = Path(__file__).parent
pdf_path = current_dir / "animais.pdf"

docs = PyPDFLoader(str(pdf_path)).load()

splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False).split_documents(docs)

if not splits:
    raise SystemExit("No documents to process after splitting.")

enriched = [
    Document(
        page_content=doc.page_content,
        metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)},
    )
    for doc in splits
]

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_API_MODEL", "gemini-embedding-2-preview"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True
)

store.add_documents(enriched, ids=ids)
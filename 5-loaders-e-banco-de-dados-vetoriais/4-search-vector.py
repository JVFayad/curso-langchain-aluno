import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import embeddings
from langchain_postgres import PGVector

load_dotenv()


for x in ("PGVECTOR_URL", "PGVECTOR_COLLECTION", "GOOGLE_API_KEY", "GOOGLE_API_MODEL"):
    if x not in os.environ:
        raise RuntimeError(f"Missing required environment variable: {x}")
    

query = "Tell me more about the brazilian jaguar"

embeddings  = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_API_MODEL", "gemini-embedding-2-preview"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True
)   

results = store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, start=1):
    print("="*50)
    print(f"Resultado {i} (score: {score:.2f}):")
    print("="*50)

    print("\nTexto:\n")
    print(doc.page_content.strip())


print("\nMetadados:\n")
for key, value in doc.metadata.items():
    print(f"{key}: {value}")

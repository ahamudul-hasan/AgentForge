import os
import time
from dotenv import load_dotenv

from langsmith import traceable  # <-- key import

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGSMITH_PROJECT'] = 'RAG-V2-test-2'

load_dotenv()

# Resolve PDF and cache paths relative to this script or current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(SCRIPT_DIR, "islr.pdf")
PDF_PATH = DEFAULT_PDF if os.path.exists(DEFAULT_PDF) else "islr.pdf"
INDEX_CACHE_DIR = os.path.join(SCRIPT_DIR, ".faiss_v2_cache")

# ---------- traced setup steps ----------
@traceable(name="load_pdf")
def load_pdf(path: str, max_pages: int | None = 20):
    loader = PyPDFLoader(path)
    docs = loader.load()  # list[Document]
    # Slicing for free-tier quotas: 441 pages produces >1,200 chunks and hits 429 quota exhaustion.
    if max_pages and len(docs) > max_pages:
        print(f"[*] Loaded {len(docs)} pages. Slicing to first {max_pages} pages to prevent free-tier 429 quota limits.")
        docs = docs[:max_pages]
    return docs

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits, batch_size: int = 25):
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = None
    total = len(splits)
    print(f"[*] Embedding {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = splits[i:i + batch_size]
        # Automatic retry with exponential backoff if 429 RESOURCE_EXHAUSTED occurs
        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, emb)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    wait_seconds = 12 * (attempt + 1)
                    print(f"[!] Rate limit (429) hit. Sleeping for {wait_seconds}s before retry (attempt {attempt + 1}/5)...")
                    time.sleep(wait_seconds)
                else:
                    raise e
        # Brief pause between batches to stay below requests-per-minute limit
        if i + batch_size < total:
            time.sleep(1.5)

    return vectorstore

# You can also trace a “setup” umbrella span if you want:
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str, force_rebuild: bool = False):
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists(INDEX_CACHE_DIR) and not force_rebuild:
        print(f"[*] Loading cached vectorstore from {INDEX_CACHE_DIR} (no embedding API calls needed)...")
        return FAISS.load_local(INDEX_CACHE_DIR, emb, allow_dangerous_deserialization=True)

    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    vs.save_local(INDEX_CACHE_DIR)
    print(f"[*] Vectorstore successfully saved to {INDEX_CACHE_DIR}")
    return vs

# ---------- pipeline ----------
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)
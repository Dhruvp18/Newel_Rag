import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY is not set in environment or .env file.")

# Loading the embedding model
embedding = HuggingFaceEmbeddings()

# Load the LLM from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Initialize BGE Reranker for two-stage retrieval
print("Loading BGE reranker model...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("BGE reranker loaded.")

# --- Anti-Gravity System Prompt ---
SYSTEM_PROMPT = """You are a professional financial analyst assistant. Your role is to analyze financial documents with precision and provide well-sourced answers.

STRICT RULES:
1. You must ONLY use the provided context documents to answer the question.
2. You must include inline citations for every claim, using the format [Page X].
3. If multiple documents support a claim, cite all relevant pages.
4. If the answer cannot be found in the context, state: "I cannot find this information in the provided document."
5. NEVER use your internal knowledge. NEVER hallucinate figures, dates, or facts.
6. When discussing financial figures, quote them exactly as they appear in the context.
7. Structure your answer clearly with paragraphs for readability.

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

Provide a thorough, well-cited answer. End your response with a "Sources:" section listing all pages referenced."""


def process_document_to_chroma_db(file_name):
    """Load and process a PDF document into ChromaDB with page-level metadata."""
    file_path = os.path.join(working_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Use PyMuPDFLoader for better table/structure extraction with page numbers
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    print(f"  Loaded {len(documents)} pages from PDF.")

    # Smaller chunks = denser information per vector for financial data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = text_splitter.split_documents(documents)

    print(f"  Split into {len(texts)} chunks.")

    # Persist to ChromaDB
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore"),
    )

    print(f"  Stored {len(texts)} chunks in ChromaDB.")
    return len(texts)


def answer_question(user_question):
    """Two-stage retrieval: over-fetch with vector search, rerank with BGE, then answer."""
    # Load the persistent vector store
    vectordb = Chroma(
        persist_directory=os.path.join(working_dir, "doc_vectorstore"),
        embedding_function=embedding,
    )

    # Stage 1: Over-fetch candidates (k=15) for reranking
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    candidate_docs = retriever.invoke(user_question)

    if not candidate_docs:
        return "No relevant documents found. Please ingest a PDF first."

    # Stage 2: Rerank with BAAI/bge-reranker-v2-m3
    pairs = [(user_question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs).tolist()

    # Handle single-doc edge case (scores may be a float instead of list)
    if isinstance(scores, (int, float)):
        scores = [scores]

    # Sort by reranker score descending, keep top 6
    scored_docs = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:6]]

    # Build labeled context with page numbers
    context_parts = []
    seen_pages = set()
    for i, doc in enumerate(top_docs, 1):
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", "?"))
        # PyMuPDFLoader uses 0-indexed pages, convert to 1-indexed for display
        if isinstance(page_num, int):
            page_num += 1
        seen_pages.add(page_num)
        context_parts.append(
            f"[Doc {i}, Page {page_num}]:\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # Format the prompt
    prompt_text = SYSTEM_PROMPT.format(
        context=context_text,
        question=user_question,
    )

    response = llm.invoke(prompt_text)
    return response.content

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

embedding = HuggingFaceEmbeddings()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


print("Loading BGE reranker model...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("BGE reranker loaded.")

SYSTEM_PROMPT = """You are a professional financial analyst assistant. Your role is to analyze financial documents with precision and provide well-sourced answers.

STRICT RULES:
1. You must ONLY use the provided context documents to answer the question.
2. You must include inline citations for every claim, using the format [Page X]. DO NOT use formats like [Doc Y, Page X].
3. If multiple pages support a claim, cite all relevant pages.
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

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    print(f"  Loaded {len(documents)} pages from PDF.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = text_splitter.split_documents(documents)

    print(f"  Split into {len(texts)} chunks.")

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore"),
    )

    print(f"  Stored {len(texts)} chunks in ChromaDB.")
    return len(texts)


def answer_question(user_question):
    """Two-stage retrieval: over-fetch with vector search, rerank with BGE, then answer."""
    vectordb = Chroma(
        persist_directory=os.path.join(working_dir, "doc_vectorstore"),
        embedding_function=embedding,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})
    candidate_docs = retriever.invoke(user_question)

    if not candidate_docs:
        return "No relevant documents found. Please ingest a PDF first."

    pairs = [(user_question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs).tolist()

    if isinstance(scores, (int, float)):
        scores = [scores]

    scored_docs = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:6]]

    # Build labeled context with page numbers ONLY
    context_parts = []
    seen_pages = set()
    for doc in top_docs:
        page_num = doc.metadata.get("page", doc.metadata.get("page_number", "?"))
        if isinstance(page_num, int):
            page_num += 1
        seen_pages.add(page_num)
        context_parts.append(
            f"[Page {page_num}]:\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    prompt_text = SYSTEM_PROMPT.format(
        context=context_text,
        question=user_question,
    )

    response = llm.invoke(prompt_text)
    return response.content

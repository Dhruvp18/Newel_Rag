import os
import argparse
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(working_dir, "doc_vectorstore")

def main():
    parser = argparse.ArgumentParser(description="Local CLI for RAG")
    parser.add_argument("--ingest", type=str, help="Path or name of PDF file to ingest")
    parser.add_argument("--force-ingest", action="store_true", help="Force re-ingestion even if vectorstore exists")
    
    args = parser.parse_args()
    
    if args.ingest:
        pdf_path = args.ingest
    else:
        pdf_path = "Annual-Report-FY-2023-24 (1) (1).pdf"

    # Skip ingestion if vectorstore already exists (unless --force-ingest or --ingest)
    if os.path.exists(VECTORSTORE_DIR) and not args.force_ingest and not args.ingest:
        print(f"Vectorstore already exists. Skipping ingestion of '{pdf_path}'.")
        print("  (Use --force-ingest to re-ingest, or --ingest <file> for a new PDF)")
    else:
        print(f"Ingesting document: {pdf_path}...")
        try:
            process_document_to_chroma_db(pdf_path)
            print("Document processed successfully and saved to Chroma DB.")
        except Exception as e:
            print(f"Failed to process {pdf_path}. Error: {e}")
        
    # By default we start interactive QA loop, or if --ask is given
    print("\nType your question (or 'quit', 'exit' to leave):")
    while True:
        try:
            question = input("\nQ: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question.strip():
                continue
            
            print("Thinking...")
            answer = answer_question(question)
            print(f"\nA: {answer}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
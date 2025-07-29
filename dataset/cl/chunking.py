import os
import fitz  
import json
from  .cleaner import remove_abstract_and_intro
from .chunker_wrapper  import chunk_documents

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")
        return ""

def main(pdf_folder="pdfs", output_json="chunked_docs.json"):
    all_texts = []
    filenames = []

    for filename in os.listdir(pdf_folder):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_folder, filename)
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue

        cleaned = remove_abstract_and_intro(text)
        all_texts.append(cleaned)
        filenames.append(filename)

    
    chunked = chunk_documents(all_texts)

    
    data = {}
    for name, chunks in zip(filenames, chunked):
        data[name] = [c.text for c in chunks]

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} documents to {output_json}")

if __name__ == "__main__":
    main()

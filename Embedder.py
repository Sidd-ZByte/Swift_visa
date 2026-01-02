
# 1. IMPORT DEPENDENCIES


from pypdf import PdfReader
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import pickle


# ENABLE LOGGING


logging.basicConfig(
    filename="process_log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)




# LOAD PDF & EXTRACT TEXT


def load_multiple_pdfs(paths):
    all_text = ""
    
    for path in paths:
        print(f"Loading: {path}")
        logging.info(f"Loading PDF: {path}")
        
        reader = PdfReader(path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            print(f"  Extracting Page {page_num+1}...")
            logging.info(f"Extracting Page {page_num+1} from {path}")
            
            extracted = page.extract_text()
            if extracted:
                text += extracted
        
        print(f"  ✔ Finished {path}, length = {len(text)} characters\n")
        logging.info(f"Completed {path} | Extracted length = {len(text)} characters")
        
        all_text += text + "\n"
    
    print("All PDFs loaded successfully!\n")
    logging.info("All PDFs loaded successfully!")
    
    return all_text




# PREPROCESS TEXT


def preprocess(text):
    logging.info("Starting preprocessing...")
    
    orig_len = len(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,? ]', '', text)
    
    logging.info(f"Preprocessing completed | Original length: {orig_len}, Clean length: {len(text)}")
    return text




# CHUNK TEXT


def chunk_text(text, chunk_size=500, overlap=100):
    logging.info(f"Chunking text | Chunk size={chunk_size}, Overlap={overlap}")
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

def save_chunks(chunks, output_file="chunks.pkl"):
    logging.info(f"Saving chunks to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    logging.info(f"Chunks saved successfully | File: {output_file}")


# GENERATING EMBEDDINGS


def generate_embeddings(chunks):
    logging.info("Loading embedding model: all-MiniLM-L6-v2")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    
    logging.info(f"Embeddings created | Total vectors: {len(embeddings)}, Dimension: {len(embeddings[0])}")
    return embeddings

def save_embeddings_matrix(embeddings, output_file="embeddings.npy"):
    logging.info(f"Saving embeddings matrix to {output_file}...")
    np.save(output_file, embeddings)
    logging.info(f"Embeddings matrix saved successfully | Shape: {np.array(embeddings).shape}")



# STORING IN FAISS


def save_faiss_index(embeddings, output_file="visa_index.faiss"):
    logging.info("Creating FAISS index...")
    
    embedding_array = np.array(embeddings).astype("float32")
    dim = embedding_array.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embedding_array)
    faiss.write_index(index, output_file)

    logging.info(f"FAISS index saved | File: {output_file}, Dimension: {dim}, Vectors: {len(embeddings)}")




if __name__ == "__main__":
    
    print("\nSTEP 1: Loading PDFs...")
    logging.info("STEP 1: Loading PDFs started")

    pdf_files = [
        "UK_eligible.pdf",
        "USA_visa_eli.pdf",
        "Canada_eligible.pdf",
        "Schengen_visa.pdf",
        "ireland_visa.pdf"
    ]

    text = load_multiple_pdfs(pdf_files)
    print("All PDFs Loaded!\n")
    logging.info("All PDFs Loaded")



    # ---- PREPROCESS ----
    print("\nSTEP 2: Preprocessing Text")
    logging.info("STEP 2: Preprocessing started")

    clean_text = preprocess(text)
    print("Clean text length:", len(clean_text))
    print("Clean text preview:\n", clean_text[:500], "\n")



    # ---- CHUNKING ----
    print("\nSTEP 3: Chunking Text")
    logging.info("STEP 3: Chunking started")

    chunks = chunk_text(clean_text)
    print("Total chunks created:", len(chunks))
    print("First chunk preview:\n", chunks[0][:300], "\n")

    print("\nSTEP 3b: Saving chunks to chunks.pkl")
    save_chunks(chunks)


    # ---- EMBEDDINGS ----
    print("\nSTEP 4: Generating Embeddings")
    logging.info("STEP 4: Embeddings started")

    embeddings = generate_embeddings(chunks)
    print("Total embeddings created:", len(embeddings))
    print("Embedding vector (first 10 numbers):\n", embeddings[0][:10], "\n")
   
    print("Embeddings matrix shape:", np.array(embeddings).shape)

    print("\nSTEP 4b: Saving embeddings matrix")
    save_embeddings_matrix(embeddings)

    # ---- SAVE FAISS ----
    print("\nSTEP 5: Saving FAISS Index")
    logging.info("STEP 5: Saving FAISS index started")

    save_faiss_index(embeddings)
    print("FAISS index saved as visa_index.faiss\n")
    logging.info("FAISS index saved successfully")
    # ---- SAVE CHUNKS ----

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("Chunks saved as chunks.pkl")
logging.info("Chunks saved successfully")
save_faiss_index(embeddings, output_file="faiss_index.bin")
print("FAISS index saved as faiss_index.bin")

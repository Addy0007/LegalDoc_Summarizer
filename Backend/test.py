import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
print("🔁 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded and using device: {device}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    print(f"📂 Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()
        if page_text:
            text += page_text + "\n"
        print(f"   - Extracted page {page_num + 1}")
    return text

# Function to chunk text for the model
def chunk_text(text, tokenizer, max_tokens=1024):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokenized_len = len(tokenizer(" ".join(current_chunk), return_tensors="pt")["input_ids"][0])
        if tokenized_len > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"🧩 Text split into {len(chunks)} chunk(s)")
    return chunks

# ✅ MAIN FUNCTION TO BE CALLED FROM FLASK
def summarize_pdf(pdf_path):
    print("⚙️ Summarizing PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    
    if not full_text.strip():
        raise ValueError("The PDF contains no readable text.")

    chunks = chunk_text(full_text, tokenizer)

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"📝 Summarizing chunk {i + 1} of {len(chunks)}...")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(device)

        summary_ids = model.generate(
            input_ids,
            max_length=500,
            min_length=100,
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summaries.append(summary)

    final_input = " ".join(summaries)
    print("🔁 Running final summary aggregation...")
    final_inputs = tokenizer(final_input, return_tensors="pt", truncation=True, max_length=1024)
    final_input_ids = final_inputs["input_ids"].to(device)

    final_summary_ids = model.generate(
        final_input_ids,
        max_length=1000,
        min_length=200,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True
    )

    final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("✅ Final summary complete.")
    return final_summary

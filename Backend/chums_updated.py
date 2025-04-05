import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ✅ Load local model
model_dir = "/home/dell/Desktop/Major/model-dir"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# ✅ Function to read text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✅ Function to chunk text
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

    return chunks

# ✅ Read and chunk PDF text
pdf_path = '/home/dell/Desktop/Major/Backend/sample.pdf'
full_text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(full_text, tokenizer)

# ✅ Generate summaries for each chunk
summaries = []

for chunk in chunks:
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

# ✅ Final summarization
final_input = " ".join(summaries)
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

# ✅ Print result
print("\n📝 Final Summary:\n", final_summary)

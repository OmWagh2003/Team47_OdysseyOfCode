import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from section_utils import extract_sections

# ========== CONFIG ==========
DATA_FOLDER = "data"
COMPANY_DATA_FILE = os.path.join(DATA_FOLDER, "company_data.txt")

# ========== LOAD MODEL ==========
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ========== UTILITY FUNCTIONS ==========
def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def count_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=1800):
    tokens = tokenizer.encode(text)
    return [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

def call_local_model(prompt, chunking=True):
    outputs = []

    if chunking and count_tokens(prompt) > 2048:
        chunks = chunk_text(prompt)
        for chunk in chunks:
            input_ids = torch.tensor([chunk]).to(model.device)
            attention_mask = (input_ids != tokenizer.eos_token_id).long()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=300,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(decoded)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return "\n\n".join(outputs)

# ========== ANALYSIS FUNCTION ==========
def analyze_risk(company_data: str, rfp_text: str, rfp_filename: str = "") -> str:
    sections = extract_sections(rfp_text)
    risk_text = sections.get("Risk Analysis", "")

    risk_prompt = f"""
You are a contract analyst.

Identify any risks, ambiguities, or unusual clauses in the terms and evaluation process.

Risk Section:
{risk_text}
"""
    risk_response = call_local_model(risk_prompt)

    if rfp_filename:
        print(f"\n📂 Analyzing {rfp_filename}...\n{'='*60}")
        print("\n📌 Risk Analysis:\n", risk_response)

    return risk_response

# ========== MAIN EXECUTION FOR DEBUGGING ==========
if __name__ == "__main__":
    RFP_FILES = [
        "ELIGIBLE RFP - 1.txt",
        "ELIGIBLE RFP - 2.txt",
        "IN-ELIGIBLE_RFP.txt",
    ]

    company_data = read_file(COMPANY_DATA_FILE)

    for rfp_filename in RFP_FILES:
        rfp_path = os.path.join(DATA_FOLDER, rfp_filename)
        rfp_text = read_file(rfp_path)
        analyze_risk(company_data, rfp_text, rfp_filename)

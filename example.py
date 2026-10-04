"""The README's MicroGlot and MicroGlot-plain examples as one script: python example.py

The first run downloads both models (about 12 GB).
"""
import torch
from transformers import AutoModel, AutoTokenizer

# MicroGlot: DNA and its species
tokenizer = AutoTokenizer.from_pretrained("athanzli/MicroGlot", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "athanzli/MicroGlot", trust_remote_code=True, dtype=torch.bfloat16
).to("cuda")

sequences = ["ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCC", "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC"]
species = ["Escherichia coli", "Bacillus subtilis"]

inputs = tokenizer(sequences, species=species, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_layer = outputs.last_hidden_state   # [2, length, 1024]
print("MicroGlot:      ", len(outputs.hidden_states), "hidden states of shape", tuple(last_layer.shape))

del model, outputs
torch.cuda.empty_cache()

# MicroGlot-plain: DNA only
tokenizer = AutoTokenizer.from_pretrained("athanzli/MicroGlot", subfolder="plain", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "athanzli/MicroGlot", subfolder="plain", trust_remote_code=True, dtype=torch.bfloat16
).to("cuda")

sequences = ["ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCC", "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGC"]

inputs = tokenizer(sequences, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_layer = outputs.last_hidden_state   # [2, length, 1024]
print("MicroGlot-plain:", len(outputs.hidden_states), "hidden states of shape", tuple(last_layer.shape))

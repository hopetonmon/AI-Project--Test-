from transformers import pipeline  # Imports Hugging Face’s pipeline tool

import torch  # <-- NEW: Import torch to detect MPS (Mac GPU)

# --- NEW: select device ---
device = 0 if torch.backends.mps.is_available() else -1  
# Hugging Face uses: 0 = GPU, -1 = CPU

# Creates a summarization pipeline and specifies device
model = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=device  # <-- ADDED: tells pipeline to use MPS GPU if available
)

# Feeds the text "text to summarize" into the model
response = model("text to summarize")

# Prints the summarized text to the console
print(response)

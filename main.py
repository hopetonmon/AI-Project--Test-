from transformers import pipeline  # Imports Hugging Face’s pipeline tool

import torch  # <-- NEW: Import torch to detect MPS (Mac GPU)

device = 0 if torch.backends.mps.is_available() else -1  #command to run on GPU if available, else CPU.
# Hugging Face uses: 0 = GPU, -1 = CPU

# Creates a summarization pipeline and specifies device
model = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=device  # <-- ADDED: tells pipeline to use MPS GPU if available
)

# Feeds the text "text to summarize" into the model
response = model("Artificial intelligence (AI) is transforming nearly every industry, from healthcare to finance, by automating tasks, improving efficiency, and enabling new insights. In healthcare, AI algorithms can analyze medical images and predict patient outcomes more accurately than traditional methods. In finance, AI systems detect fraud and optimize investment strategies. While the benefits are significant, AI also raises ethical and societal concerns, such as job displacement, privacy issues, and potential biases in decision-making algorithms. Policymakers and organizations must carefully balance innovation with responsible use to ensure AI technologies benefit society as a whole.", max_length=50, min_length=25, do_sample=False)

# Prints the summarized text to the console
print(response)

# Use a pipeline as a high-level helper
from transformers import pipeline

model = pipeline("any-to-any", model = "facebook/bart-large-mnli")
text = "The latest press release details the company's new policy on remote work, including guidelines for team communication and hardware allocation for employees worldwide."

labels = ["Employee Relations", "Financial News", "Product Announcement", "Technical Support"]
result = model(text, labels)
print("Zero Classification Results")
print(f"input text: {result['sequence']}")
print("classification scores:")

for i,(label, score) in enumerate(zip(labels,result['scores'])):
    print(f"  {i+1}. {label}: {round(score * 100, 2)}%")
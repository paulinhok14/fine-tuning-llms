import random
import json
import os

# Import glossary from separate module
from glossary import glossary

# Output path for dataset
output_file = "data/training_dataset.jsonl"

# Generate Q&A pairs from glossary
def generate_qa(glossary, n=100):
    qa_list = []
    chosen = random.choices(glossary, k=n)
    for term, explanation in chosen:
        question = f"What does '{term}' mean in the Embraer context?"
        answer = explanation
        qa_list.append({"prompt": question, "response": answer})
    return qa_list

# Generate dataset
generated_qa = generate_qa(glossary, 100)

# Append specific test questions
test_questions = [
    {
        "prompt": "What does ATP mean and why is it important in materials planning?",
        "response": "ATP stands for 'Available to Promise'. It represents physical stock minus overdue orders, essential for delivery planning."
    },
    {
        "prompt": "What's the difference between PO and POi in Embraer's SAP system?",
        "response": "PO is an external Purchase Order from a customer, while POi is an internal order for transfers between Embraer sites."
    },
    {
        "prompt": "How does the AHEAD system support aircraft maintenance?",
        "response": "AHEAD monitors the aircraft’s health and detects failures mid-flight, enabling proactive maintenance."
    },
    {
        "prompt": "What is EEC at Embraer?",
        "response": "EEC stands for Embraer Executive Care, a warranty extension program for executive jets."
    },
    {
        "prompt": "How is the Pareto Principle applied in Embraer's planning process?",
        "response": "According to Pareto, 80% of effects stem from 20% of causes, guiding prioritization of material and service planning."
    }
]

# Add them to the final list
generated_qa.extend(test_questions)

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(json.dumps(qa, ensure_ascii=False) for qa in generated_qa) + "\n")

print(f"Dataset saved to {output_file} with {len(generated_qa)} entries.")

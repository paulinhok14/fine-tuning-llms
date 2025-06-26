import random
import json
import os

# Output path to save the JSONL data to a file
output_file = "data/training_dataset.jsonl"  # Relative to the fine-tuning-llms directory


# Simplified list of themes detected in the text
themes = [
    "Aviação Executiva", "Aviação Comercial", "Modelos de Aeronaves", "Siglas e Manuais",
    "Processos de Planejamento", "Requisição de Compra", "KPIs", "Centros de Distribuição",
    "SAP (Telas e Transações)", "Boletins de Serviço", "Peças Reparáveis e Expendables",
    "Agências Certificadoras", "Conceitos Lean e Six Sigma", "Part Numbers", "Prazos e Lead Time",
    "Ferramentas de Qualidade", "Logística e Supply Chain", "E-mails e Contatos", "Documentos e Tipos de Ordem"
]

# Function to generate questions and answers
def generate_qa(themes, n=100):
    qa_list = []
    for _ in range(n):
        theme = random.choice(themes)
        question = f"O que é relacionado ao tema '{theme}'?"
        answer = f"O tema '{theme}' está relacionado a operações, processos ou elementos dentro da Embraer que envolvem {theme.lower()}."
        qa_list.append({"prompt": question, "response": answer})
    return qa_list

# Generate QAs
generated_qa = generate_qa(themes, 100)

# Generate 5 specific test questions
test_questions = [
    {"prompt": "O que significa ATP e qual sua importância no planejamento de materiais?", 
     "response": "ATP significa 'Available to Promise'. Ele representa o estoque físico menos os pedidos em atraso, sendo essencial para prometer entregas ao cliente."},
    {"prompt": "Qual a diferença entre PO e POi no SAP da Embraer?", 
     "response": "PO é uma 'Purchase Order' de cliente, enquanto POi é uma ordem de compra interna para transferências entre sites da Embraer."},
    {"prompt": "Como o sistema AHEAD contribui para a manutenção das aeronaves?", 
     "response": "O AHEAD é um sistema de diagnóstico de saúde da aeronave que permite detectar falhas em voo, otimizando o tempo de resposta de manutenção."},
    {"prompt": "O que é o EEC na Embraer?", 
     "response": "EEC é o 'Embraer Executive Care', uma extensão de garantia para aeronaves executivas."},
    {"prompt": "Como o conceito de Pareto é aplicado no planejamento da Embraer?", 
     "response": "Segundo o princípio de Pareto, 80% dos impactos vêm de 20% das causas, guiando decisões de priorização no planejamento."}
]

# Add test questions to the end
generated_qa.extend(test_questions)

# Generate JSONL content
jsonl_content = "\n".join(json.dumps(qa, ensure_ascii=False) for qa in generated_qa)

# Ensure the data directory exists
data_dir = os.path.dirname(output_file)
os.makedirs(data_dir, exist_ok=True)

# Save the JSONL data to a file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(jsonl_content + "\n")  # Add newline at the end of file

print(f"Data saved to {output_file} with {len(generated_qa)} questions")

import random
import json

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
        qa_list.append({"pergunta": question, "resposta": answer})
    return qa_list

# Generate QAs
generated_qa = generate_qa(themes, 100)

# Generate 5 specific test questions
test_questions = [
    {"pergunta": "O que significa ATP e qual sua importância no planejamento de materiais?", 
     "resposta": "ATP significa 'Available to Promise'. Ele representa o estoque físico menos os pedidos em atraso, sendo essencial para prometer entregas ao cliente."},
    {"pergunta": "Qual a diferença entre PO e POi no SAP da Embraer?", 
     "resposta": "PO é uma 'Purchase Order' de cliente, enquanto POi é uma ordem de compra interna para transferências entre sites da Embraer."},
    {"pergunta": "Como o sistema AHEAD contribui para a manutenção das aeronaves?", 
     "resposta": "O AHEAD é um sistema de diagnóstico de saúde da aeronave que permite detectar falhas em voo, otimizando o tempo de resposta de manutenção."},
    {"pergunta": "O que é o EEC na Embraer?", 
     "resposta": "EEC é o 'Embraer Executive Care', uma extensão de garantia para aeronaves executivas."},
    {"pergunta": "Como o conceito de Pareto é aplicado no planejamento da Embraer?", 
     "resposta": "Segundo o princípio de Pareto, 80% dos impactos vêm de 20% das causas, guiando decisões de priorização no planejamento."}
]

# Add test questions to the end
generated_qa.extend(test_questions)

# Return formatted JSONL data
jsonl_content = "\n".join(json.dumps(qa, ensure_ascii=False) for qa in generated_qa)
jsonl_content[:1000]  # Return only the beginning for preliminary visualization

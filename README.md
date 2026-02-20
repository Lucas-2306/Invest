# 📈 Quant Directional Model - Brasil

Projeto de análise quantitativa com Machine Learning para prever a **direção futura (alta/queda)** de ações da B3 no horizonte **semanal ou mensal**.

---

## 🎯 Objetivo

Construir um modelo de classificação binária que prevê:

- **1 → Ação deve subir**
- **0 → Ação deve cair ou ficar estável**

Horizontes suportados:

- 📅 Semanal (~5 pregões)
- 📆 Mensal (~21 pregões)

---

## 🏗 Estrutura do Projeto

```
quant_brasil/
│
├── data/
│   ├── raw/          # Dados brutos baixados da API
│   └── processed/    # Dados tratados com features
│
├── notebooks/        # Exploração e experimentos
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── backtest.py
│   └── utils.py
│
├── .env              # Variáveis de ambiente (não versionado)
├── .gitignore
└── README.md
```

---

## 📊 Fonte de Dados

Os dados de mercado são obtidos via API da brapi.dev, contendo:

- Open
- High
- Low
- Close
- Volume
- Adjusted Close

Os dados são salvos em:

```
data/raw/
```

---

## 🔐 Configuração da API Key

Para acessar a brapi.dev é necessário criar uma conta e gerar uma **API Key**.

### 🔹 Forma recomendada (profissional)

Criar um arquivo `.env` na raiz do projeto:

```
BRAPI_API_KEY=SUA_CHAVE_AQUI
```

Esse arquivo está listado no `.gitignore`, portanto **não será enviado ao GitHub**.

---

### 🔹 Carregamento da chave no projeto

O projeto usa `python-dotenv` para carregar a chave automaticamente:

```python
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("BRAPI_API_KEY")
```

---

### ⚠️ Importante sobre segurança

- ❌ Nunca coloque a chave diretamente no código
- ❌ Nunca envie a chave para o GitHub
- ✅ Sempre use variável de ambiente ou `.env`
- ✅ Verifique com `git status` antes de fazer commit

---

### 🔎 Se você usou `export`

Se você executou:

```bash
export BRAPI_API_KEY="SUA_CHAVE"
```

A variável:
- Fica ativa apenas na sessão atual do terminal
- É apagada quando o terminal é fechado
- Não fica salva em nenhum arquivo do sistema

---

## 🚀 Como Executar

1️⃣ Criar ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

2️⃣ Instalar dependências:

```bash
pip install -r requirements.txt
```

3️⃣ Configurar `.env` com sua API key

4️⃣ Rodar coleta de dados:

```bash
python -m src.data_loader
```

---

## 🧠 Pipeline do Projeto

1. Coleta de dados  
2. Criação de features técnicas  
3. Construção do target semanal/mensal  
4. Split temporal  
5. Treinamento do modelo  
6. Backtest da estratégia  
7. Avaliação financeira (Sharpe, drawdown, retorno acumulado)  

---

## 📌 Próximos Passos

- Implementar `features.py`
- Construir modelo base (Logistic Regression / XGBoost)
- Criar backtest simples
- Comparar contra benchmark (IBOV)

---

## 📄 Licença

Projeto para fins educacionais e de pesquisa.

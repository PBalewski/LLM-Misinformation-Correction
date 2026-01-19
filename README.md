# 🇪🇸 MUSE Model Reproduction on Spanish Fake News Corpus

This repository reproduces the **MUSE misinformation correction model**
introduced in **“MUSE: Multi-Stage Explanation-Based Correction of Misinformation”**
(arXiv:2403.11169), extending the original work to a **cross-lingual Spanish setting**
using the **Spanish Fake News Corpus**.

The goal of this project is to evaluate whether the MUSE framework generalizes
beyond English and to analyze its effectiveness for misinformation correction
in Spanish-language news.

---

## 📄 Original Paper

> **MUSE: Multi-Stage Explanation-Based Correction of Misinformation**  
> arXiv:2403.11169  
> https://arxiv.org/abs/2403.11169

This project is a **reproduction and extension**, not the original implementation.

---

## 🔍 Project Objectives

- Reproduce the MUSE pipeline as described in the original paper
- Adapt the methodology to a **Spanish-language misinformation dataset**
- Analyze cross-lingual robustness of explanation-based correction
- Document challenges related to multilingual prompting and evaluation

---

## 🧠 MUSE Algorithm Pipeline

The following diagram illustrates the original MUSE pipeline:

![MUSE pipeline](https://github.com/PBalewski/LLM-Misinformation-Correction/blob/main/images/pipeline.jpg)

**Key stages:**
1. Claim identification  
2. Explanation generation  
3. Explanation-guided correction  
4. Final misinformation correction  

---

## 📊 Dataset

**Spanish Fake News Corpus**

- Language: Spanish  
- Domain: News articles and claims  
- Task: Misinformation detection and correction  

The dataset is used to evaluate the cross-lingual transferability of the MUSE
framework.

---

## ⚙️ Status of Work

Current progress:

- [x] Literature review and methodology analysis  
- [x] Dataset selection (Spanish Fake News Corpus)  
- [x] Final model selection and justification  
- [x] Toolchain and environment definition  
- [x] Data preprocessing and normalization  
- [x] MUSE pipeline implementation  
- [x] Evaluation and error analysis  
- [x] Technical report  

---

## 🛠️ Toolchain & Environment

- Python 3.10-3.12 
- Hugging Face Transformers  
- PyTorch  
- spaCy (Spanish models)  
- NumPy / Pandas  
- Jupyter Notebook  

---

## ▶️ How to Run the Project

```bash
git clone https://github.com/PBalewski/LLM-Misinformation-Correction.git
cd LLM-Misinformation-Correction
pip install -r requirements.txt
```

Some experiments require API-based LLMs.

Create a file:
```bash
model/data/api_keys.json
```
Make sure following structure is preserved:
```bash
{
    "gemini": "API_KEY",
    "GoogleSearch": {
        "Key": "API_KEY",
        "High-Priority": "GOOGLE_PROGRAMMABLE_SEARCH_ID",
        "Medium-Priority": "GOOGLE_PROGRAMMABLE_SEARCH_ID",
        "Low-Priority": "GOOGLE_PROGRAMMABLE_SEARCH_ID"
}
```

Do not commit API keys to the repository.

---

## 📁 Repository Structure

```
.
├── docs/                 # PDFs and documentation
├── images/               # Figures and diagrams
├── model/                # LLM wrappers and prompting logic
├── notebooks/            # Exploratory data analysis and experiments
├── README.md
├── requirements.txt      # Python dependencies
└── TECHNICAL_REPORT.md   # Technical report of the project
```

---

## 📈 Evaluation

Evaluation will follow the original paper where possible, including:

- Correction accuracy  
- Explanation usefulness  
- Error typology analysis  

Adaptations required for Spanish-language evaluation will be explicitly documented.

---

## ⚠️ Limitations

- Results may differ due to multilingual prompting effects  
- Dataset annotation standards differ from the original English benchmarks  
- LLM behavior may vary across model versions
- Most LLMs have limited number of tokens to use per day/week/month 

---

## 📜 License

This repository is released for **research and educational purposes only**.

Original paper concepts and methodology remain the intellectual property of their
respective authors.

---

## ✍️ Authors

- **Piotr Balewski** - [@PBalewski](https://github.com/PBalewski)
- **Kuba Czech** — [@KubaCzech](https://github.com/KubaCzech)
- **Vasyl Korzavatykh** — [@vasia-korz](https://github.com/vasia-korz)
- **Wojciech Nagórka** — [@WojtekNagorka](https://github.com/WojtekNagorka)

---

## 📌 Disclaimer

This project is **not affiliated with nor endorsed by the original MUSE authors**.
It is an independent academic reproduction aimed at evaluating cross-lingual
generalization.
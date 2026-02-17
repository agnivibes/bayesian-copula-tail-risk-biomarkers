# Bayesian Inference for Joint Tail Risk in Paired Biomarkers via Archimedean Copulas 🧬📈

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Bayesian](https://img.shields.io/badge/Bayesian-Restricted%20Jeffreys%20Prior-purple)]()
[![Copulas](https://img.shields.io/badge/Copulas-Clayton%2FGumbel-orange)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
[![Dataset](https://img.shields.io/badge/Data-NHANES%202017--2018-yellowgreen)](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2017-2018)

This repository accompanies the paper **“Bayesian Inference for Joint Tail Risk in Paired Biomarkers via Archimedean Copulas with Restricted Jeffreys Priors.”**  
We provide a fully reproducible pipeline for Bayesian copula inference of clinically interpretable tail-risk summaries using **Clayton** and **Gumbel** Archimedean copulas.

🔍 **What’s in this repo:**
- **Script 1 (data prep):** downloads/reads the NHANES 2017–2018 laboratory XPT files  
  **`GLU_J.XPT`** (Plasma Fasting Glucose) and **`GHB_J.XPT`** (Glycohemoglobin), merges them by **`SEQN`**, and exports the analysis-ready CSV.
- **Script 2 (analysis):** runs the Bayesian restricted-Jeffreys posterior inference on pseudo-observations and reproduces the paper’s main tables and figures (including posterior tail-risk plots).

📌 The NHANES laboratory files can be accessed from the CDC portal (2017–2018, Laboratory component):  
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2017-2018

---

## 📦 Requirements
Install dependencies:
```bash
pip install numpy pandas scipy matplotlib pyreadstat
```

## 🚀 Getting Started
```bash
git clone https://github.com/agnivibes/bayesian-copula-tail-risk-biomarkers.git
cd bayesian-copula-tail-risk-biomarkers
```

## 🔬 Research Paper

Aich, A., Murshed, M.M., Hewage, S., Aich, A.B (2026). Bayesian Inference for Joint Tail Risk in Paired Biomarkers via Archimedean Copulas with Restricted Jeffreys Priors.[Manuscript under review]

## 📊 Citation
If you use this code or method in your own work, please cite:

@article{Aich2026BayesianCopulaTailRisk,
  title  = {Bayesian Copula-Based Tail Risk Inference for Paired Biomarkers (NHANES 2017--2018)},
  author = {Aich, Agnideep and Murshed, Md Monzur and Hewage, Sameera and Aich, Ashit Baran},,
  year   = {2026},
  note   = {Manuscript under review}
}

## 📬 Contact
For questions or collaborations, feel free to contact:

Agnideep Aich,
Department of Mathematics, University of Louisiana at Lafayette
📧 agnideep.aich1@louisiana.edu

## 📝 License
This project is licensed under the [MIT License](LICENSE).

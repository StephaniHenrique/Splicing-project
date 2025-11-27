# README

## Overview

This repository contains **two independent projects**:

1. **Cognate Sequence Prediction Model**  
2. **Model Distillation Pipeline**

Each project has its own directory structure and execution flow. Follow the instructions below to correctly set up the environment, prepare the data, and run the models.

---

## 1. Creating a Virtual Environment

Before running any of the projects, create and activate a Python virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Project 1 — Cognate Sequence Prediction

### How to Run

1. Navigate to the project folder:
   ```bash
   cd cognate_sequence_prediction/Code_V1
   ```

2. **Run the data processing script first:**
   ```bash
   python data.py
   ```

3. **Then run the model script:**
   ```bash
   python model.py
   ```

---

## 3. Project 2 — Model Distillation

### Required Downloads

Before running the code, download the following files:

- **Human reference genome (FASTA)**
- **Human gene annotation (GTF)**

### How to Download via Terminal

```bash
# Download genome FASTA
wget https://ftp.ensembl.org/pub/release-87/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

# Download GTF annotation
wget https://ftp.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh38.87.gtf.gz

# Unzip files
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.87.gtf.gz
```

### How to Run

1. Navigate to the project folder:
   ```bash
   cd model_distillation/Code
   ```

2. **Construct the ENSEMBLE dataset:**
   ```bash
   construct_ENSMBL_datasets.ipynb
   ```
3. Navigate to src folder:
   ```bash
   cd src
   ```
   
4. **Run the distillation script:**
   ```bash
   python model_distillation.py
   ```

---

## Final Notes

- Always double-check the paths before running the scripts, especially if moving files between directories!


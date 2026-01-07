# CMSH
python pytorch
1. Overview
CMHS is a deep learning framework for MHC–peptide binding prediction
The repository supports direct reproduction of the main experimental results reported in the paper using provided pretrained model weights.
#############################################
2. Repository Structure
CMSH/
├── data/
│   ├── mhc/                # Original structure datasets
│   └── HLA_all.txt          # Original sequence datasets
│
├── models/
│   ├── esm_lora_cmsh.py                # Full CMHS model definition
│   └── lora_cmsh_test.py             # Full CMHS model definition
│
├── ESM3/
│   ├── esm3_mhc.py        # structural data presiction
│
├── scripts/
│   ├── main_g.py
│   └── main_all.py
│
├── README.md
#############################################
3. Environment Setup
3.1 Requirements
Python ≥ 3.8;
PyTorch ≥ 1.12;
CUDA 
#############################################
4. Data Preparation
4.1 Dataset
The structure data calculated by ESM3:
python ESM3/esm3_mhc.py
#############################################
5. Running Experiments
5.1 Training from Scratch
python scripts/main_g.py \
This will:
Fine-tune ESM-2 using LoRA
Train CMSH modules
Save checkpoints and logs
##############################################
5.2 Evaluation with Pretrained Models (Recommended)
To directly reproduce the reported results:
python scripts/main_all.py \

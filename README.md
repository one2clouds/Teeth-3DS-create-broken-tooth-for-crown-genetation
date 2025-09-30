# Dataset-Preparation-For-Reconstruction-Generation-for-Teeth3DS

## Download Dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/accelerated-komputing/CROWN_GEN_DATASET
```

## Installation
```bash
pip install numpy open3d torch torchvision
```

## Run scripts 
### ITERO Context Surface & Crown Extraction
```bash 
python context_surface_and_crown_extraction_ITERO.py
```

### Partial Context Surface & Crown Extraction
```bash
python context_surface_and_crown_extraction_PARTIAL.py
```

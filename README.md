# Dataset-Preparation-For-Reconstruction-Generation-for-Teeth3DS

## Download Dataset

```bash
pip install huggingface_hub
huggingface-cli download accelerated-komputing/CROWN_GEN_DATASET --repo-type dataset --local-dir CROWN_GEN_DATASET
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

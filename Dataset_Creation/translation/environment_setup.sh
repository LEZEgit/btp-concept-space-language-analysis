# This environment is specifically for translation purposes, because the models we are using for translation needs an older version of hugging face transformer. 
# Downgrading the transformer version in venv (from `conda activate venv`) environment will create version mismatch with other files. 


# 1. Create a fresh environment with Python and ipykernel
conda create -n indic_trans python=3.10 ipykernel -y

# 2. Activate the new isolated environment
conda activate indic_trans

# Install PyTorch (adjust cuda version if needed, or use CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install pinned transformers (<4.39.0) and dependencies
pip install "transformers>=4.33.0,<4.39.0" sentencepiece accelerate huggingface_hub pandas tqdm

# Install the toolkit
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

# Register the Environment as a Jupyter Kernel
python -m ipykernel install --user --name indic_trans --display-name "Python (IndicTrans)"

cd
export DEBIAN_FRONTEND=noninteractiveS
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11
sudo apt install -y python3.11-venv python3.11-distutils 
sudo apt install -y libpython3.11-dev python3.11-dev

python3.11 -m ensurepip
python3.11 -m venv py311cuda
source ~/py311cuda/bin/activate


git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git
cd Cosmos-Tokenizer
apt-get install -y ffmpeg
pip install -e .

# Installing latest PyTorch and other packages
pip install --upgrade pip
pip install torch tqdm click transformers wandb plotly pandas datasets
wandb login
# AI MUSIC

You can sing any song using the DiffSinger(https://github.com/openvpi/DiffSinger) model.

Not only can it sing and recognize lyrics, but it also supports modifying lyrics creatively.


## Installation

```
conda create -n music python=3.8 -y
conda activate music
```

1.git clone https://github.com/hunkunai/music.git
2.conda create -n diffsinger python=3.8 -y
3.pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
4.pip install -r requirements.txt

pgvector montreal-forced-aligner=2.0.6 pynini hdbscan librosa-0.8.1
conda install -c conda-forge python=3.8 kaldi sox librosa biopython praatio tqdm requests colorama pyyaml pynini openfst baumwelch ngram


download models

uvr model 
path : assets/uvr5_weights/
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth

mfa model
path music/
wget https://huggingface.co/datasets/fox7005/tool/resolve/main/mfa-opencpop-extension.zip

download checkpoint 
final path checkpoint/nsf_hifigan

cd checkpoint 
wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
unzip nsf_hifigan_20221211.zip





# AI MUSIC

can sing any song you upload the audio file using the DiffSinger(https://github.com/openvpi/DiffSinger) model.

Not only can it sing and recognize lyrics, but it also supports modifying lyrics creatively.


## Installation

```
git clone https://github.com/hunkunai/music.git

cd music


conda create -n music python=3.8 -y

conda activate music


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


pip install -r requirements.txt

conda install -c conda-forge montreal-forced-aligner==2.0.6 --yes

pip install setuptools-scm

pip install pytest-runner

pip install paddlespeech==1.4.1


```


## download models

1.download checkpoint 

final path like checkpoint/nsf_hifigan,checkpoint/my_experiment
```
  cd checkpoint/
  wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
  unzip nsf_hifigan_20221211.zip
  wget https://github.com/hunkunai/music/releases/download/music/my_experiment.zip
  unzip my_experiment.zip

```

2.download uvr model 


final path like assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth

```
  mkdir assets/uvr5_weights
  cd assets/uvr5_weights/

  wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth

```

3.download mfa model 

final path like assets/uvr5_weights/mfa-opencpop-extension.zip
not to unzip that!

```

  cd assets/
  wget https://huggingface.co/datasets/fox7005/tool/resolve/main/mfa-opencpop-extension.zip

```


## User Guide

Step 1:<br/>
    start gradio ui

```
python app.py

```

Step 2:<br/>
    upload audio file and click button

    ![](https://github.com/hunkunai/music/raw/main/WechatIMG543.jpeg)








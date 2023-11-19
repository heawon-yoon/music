# AI MUSIC

可以唱任何你上传的音乐,用的是DiffSinger(https://github.com/openvpi/DiffSinger)模型.<br/>
可以上传自己的模型到checkpoint文件夹下面

不仅可以自动识别歌词,生成歌词后还支持魔改歌词.

#### 推理流程



1.音频文件uvr人声分离<br/>
2.人声降噪,增强<br/>
3.autoSliece人声按片段智能AI切片<br/>
4.ASR人声自动识别转歌词<br/>
5.生成歌词后可以手动修改歌词<br/>
6.MFA歌词音频自动强制对齐<br/>
7.数据预处理<br/>
8.开始推理<br/>
9.人声跟伴奏重新合成生成新音频<br/>

## 安装环境

```
git clone https://github.com/hunkunai/music.git

cd music


conda create -n music python=3.8 -y

conda activate music


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


pip install -r requirements.txt

conda install -c conda-forge montreal-forced-aligner==2.0.6 --yes

pip install paddlepaddle==2.4.2

pip install setuptools-scm

pip install pytest-runner

pip install paddlespeech==1.4.1


```


## 下载模型

1.下载checkpoint

最终路径: checkpoint/nsf_hifigan,checkpoint/my_experiment
```
  cd checkpoint/
  wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
  unzip nsf_hifigan_20221211.zip
  wget https://github.com/hunkunai/music/releases/download/music/my_experiment.zip
  unzip my_experiment.zip

```

2.下载 uvr 模型 


最终路径: assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth

```
  mkdir assets/uvr5_weights
  cd assets/uvr5_weights/

  wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth

```

3.下载 mfa 模型 

最终路径: assets/uvr5_weights/mfa-opencpop-extension.zip<br/>
不要解压!

```

  cd assets/
  wget https://huggingface.co/datasets/fox7005/tool/resolve/main/mfa-opencpop-extension.zip

```


## 启动UI


##### 启动跟运行过程中如果报 "XXX module not found" ,请根据requirements.txt file中的对应版本重新pip install 该模块

Step 1:<br/>
    start gradio ui

```
python app.py

```

Step 2:<br/>
    上传音频文件后点击生成歌词按钮
    <div>
      <img alt="" src="https://github.com/hunkunai/music/blob/main/WechatIMG543.jpeg" width="600" height="400" />
    <div/>



Step 3:<br/>
    点击开始推理按钮,生成最终音频
    <div>
      <img alt="" src="https://github.com/hunkunai/music/blob/main/WechatIMG544.jpeg" width="600" height="400" />
    <div/>









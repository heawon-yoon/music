import os
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)

now_dir = os.getcwd()
sys.path.append(now_dir)
from custom.slicer2 import predata
from paddlespeech.cli.asr.infer import ASRExecutor

from infer.modules.uvr5.modules import uvr

from configs.config import Config
from dotenv import load_dotenv
import torch
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
from custom.reformat_wavs import reformat_wavs
from pypinyin import pinyin, lazy_pinyin, Style
from custom.pinyin import pinyin2ph_func
from custom.enhance_tg import enhance_tg
from custom.build_dataset import build_dataset
from custom.add_ph_num import add_ph_num
from custom.estimate_midi import estimate_midi
from custom.combine_ds2 import combine_ds
from custom.path_util import get_full_path
from custom.audio import merge_wav
import librosa
import copy





logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

load_dotenv()

print("weight_uvr5_root 55555",os.getenv("weight_uvr5_root"))

config = Config()
lyc_data = []
slice_data = None
rstd = None
def gen_lyc(audio):
    
     global rstd
     global lyc_data
     tpath = random.randint(1, 100)
     rst = uvr("HP5-主旋律人声vocals+其他instrumentals","","opt2/"+tpath
     ,audio,"opt2/"+tpath,10,"wav")
     print("rst",rst)
     rstd = rst
     print("rstd",rstd)
     global slice_data
     slice_data = predata(rst["vocal"])
     print(slice_data)
     asr = ASRExecutor()
     rst_data = []
     dir_tmp_path = None
     cp_slice_data = copy.deepcopy(slice_data)
     tmp_i = 0
     for i,slice in enumerate(cp_slice_data):
         try:
             #语音转文字
             print(slice["wav_path"])
             tmp_audio = slice["wav_path"]
                
             audio, sr = librosa.load(slice["wav_path"])
             duration = librosa.get_duration(audio, sr)
             print(tmp_audio)
             print(duration)
             result = asr(audio_file=tmp_audio,force_yes=True)
             print(result)
             if result is None or result=='':
                os.remove(tmp_audio)
                del slice_data[i-tmp_i]
                tmp_i +=1
             else:
                
                lyc = {}
                lyc["length"]=len(result)
                lyc["text"]=result
                lyc_data.append(lyc)
                rst_data.append(result)

                #文字转音速.lab文件
                dir_path = os.path.dirname(tmp_audio)
                name, extension = os.path.splitext(tmp_audio)
                lab_file_path = os.path.join(dir_path, name + '.lab')
                pinyin2phs=pinyin2ph_func(result)
                print("pinyins :"+" ".join(pinyin2phs))
                with open(lab_file_path, "w") as file:
                    file.write(" ".join(pinyin2phs))
         except Exception as e:
             print("%s->Suc." % e)
     return "\n".join(rst_data)
def run_acoustic(text_input,text_speaker,audio):
     #人声分离
     #人声分离
     #rst = uvr("HP2-人声vocals+非人声instrumentals","","opt"
     #,audio,"opt",10,"wav")
     #print("rst",rst)
     print(slice_data)
     print(text_input)
     if slice_data is None:
         return "请先生成歌词"
     rst_data = []
     dir_tmp_path = None
     for i,slice in enumerate(slice_data):
         try:
             #语音转文字
             print(slice["wav_path"])
             tmp_audio = slice["wav_path"]
             dir_path = os.path.dirname(tmp_audio)
             dir_tmp_path = dir_path
                
            
         except:
             print("%s->Suc." % slice["wav_path"])
     print(dir_tmp_path)
    
     # 格式化mfa识别的音频格式
     reformat_wavs(dir_tmp_path,dir_tmp_path+"/tmp") 
    
    
     #生成mfa gridtext文件 mfa align path/to/tmp/dir/ path/to/your/dictionary.txt path/to/your/model.zip path/to/your/textgrids/ --beam 100 --clean --overwrite
     os.system("mfa align %s %s %s %s --beam 100 --clean --overwrite" % (dir_tmp_path+"/tmp", "dictionaries/opencpop.txt","mfa-opencpop-extension.zip",dir_tmp_path+"/textgrids"))
    
     #重新优化gridtext
     enhance_tg(dir_tmp_path+"/tmp","dictionaries/opencpop.txt",dir_tmp_path+"/textgrids",dir_tmp_path+"/textgrids/final")
    
     #生成transcation.csv文件
     build_dataset(dir_tmp_path+"/tmp", dir_tmp_path+"/textgrids/final", dir_tmp_path+"/tmp")
    
     #添加音高
     add_ph_num(dir_tmp_path+"/tmp/transcriptions.csv","dictionaries/opencpop.txt")
     estimate_midi(dir_tmp_path+"/tmp/transcriptions.csv",dir_tmp_path+"/tmp")
     
     #python convert_ds.py csv2ds path/to/your/transcriptions.csv path/to/your/wavs --overwrite
     os.system("python custom/convert_ds.py csv2ds %s %s " % (dir_tmp_path+"/tmp/transcriptions.csv", dir_tmp_path+"/tmp"))
    
     #生成推理文件
     ds_file = combine_ds(dir_tmp_path+"/tmp",slice_data,text_input)
      
     #开始推理   
     os.system("python scripts/infer.py acoustic %s --exp my_experiment --spk opencpop --out %s" % (ds_file,os.path.dirname(ds_file)))
     merge_wav(get_full_path(ds_file)+".wav",rstd["ins"])
     #acoustic(audio.name,'beyong','opencop','out2')
     print(rstd)
     print(get_full_path(ds_file)+".wav")
     return "\n".join(rst_data),"merged.wav"
def flip_image(x):
    return np.fliplr(x)
with gr.Blocks() as demo:
    #用markdown语法编辑输出一段话
    gr.Markdown("混鲲AI大模型")
    # 设置tab选项卡
    with gr.Tab("AI魔改翻唱"):
        #Blocks特有组件，设置所有子组件按垂直排列
        #垂直排列是默认情况，不加也没关系
        with gr.Column():
            audio = gr.File(file_count="multiple",label="上传音频", info="需上传wav格式",height="100")
            lyr_button = gr.Button("生成歌词")
            text_input = gr.Textbox(label="歌词魔改", info="一定要按原来歌词字数一样!")
            text_speaker =  gr.Dropdown(
            ["牛夫人", "Beyong", "妖姬"], label="AI原声", info="Will add more animals later!"
            )
            text_button = gr.Button("推理AI音频")
            vacal_audio = gr.Audio(label="AI音频", info="合成音频最终显示在这里")
    with gr.Tab("AI歌词"):
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")
    with gr.Tab("AI音频+视频"):
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")
    with gr.Tab("AI声音转换"):
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")
    #设置折叠内容
    with gr.Accordion("推理流程!"):
        gr.Markdown(
                 """
                1.音频文件uvr人声分离
                2.人声降噪,美化,增强
                3.autoSliece人声按片段智能AI切片
                4.ASR人声自动识别转歌词
                5.生成歌词后可以手动修改歌词
                6.MFA歌词音频自动强制对齐
                7.数据预处理
                8.开始推理
                9.人声跟伴奏重新合成生成新音频
                 """)
    text_button.click(run_acoustic, inputs=[text_input,text_speaker,audio], outputs=[text_input,vacal_audio])
    lyr_button.click(gen_lyc, inputs=[audio], outputs=[text_input])
demo.launch(server_name='172.21.182.86',server_port=7861)

import pathlib
import shutil

import click
import librosa
import numpy as np
import soundfile
import tqdm



def reformat_wavs(src, dst):
    src = pathlib.Path(src).resolve()
    dst = pathlib.Path(dst).resolve()
    print(src)
    assert src != dst, 'src and dst should not be the same path'
    assert src.is_dir() and (not dst.exists() or dst.is_dir()), 'src and dst must be directories'
    dst.mkdir(parents=True, exist_ok=True)
    samplerate = 16000
    filelist = list(src.glob('*.wav'))
    max_y = 1.0
    
    for file in tqdm.tqdm(filelist):
        y, _ = librosa.load(file, sr=samplerate, mono=True)
        soundfile.write((dst / file.name), y / max_y, samplerate, subtype='PCM_16')
        annotation = file.with_suffix('.lab')
        shutil.copy(annotation, dst)
    print('Reformatting and copying done.')



import pathlib

import click
import librosa
import numpy as np
import parselmouth as pm
import textgrid as tg
import tqdm



def enhance_tg(
        wavs, dictionary, src, dst,
        f0_min=40, f0_max=1100, br_len=0.1, br_db=-60, br_centroid=2000,
        time_step=0.005, min_space=0.04, voicing_thresh_vowel=0.45, voicing_thresh_breath=0.6, br_win_sz=0.05
):
    wavs = pathlib.Path(wavs)
    dict_path = pathlib.Path(dictionary)
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(dict_path, 'r', encoding='utf8') as f:
        rules = [ln.strip().split('\t') for ln in f.readlines()]
    dictionary = {}
    phoneme_set = set()
    for r in rules:
        phonemes = r[1].split()
        dictionary[r[0]] = phonemes
        phoneme_set.update(phonemes)

    filelist = list(wavs.glob('*.wav'))
    for wavfile in tqdm.tqdm(filelist):
        tgfile = src / wavfile.with_suffix('.TextGrid').name
        textgrid = tg.TextGrid()
        textgrid.read(str(tgfile))
        words = textgrid[0]
        phones = textgrid[1]
        


        # Remove short spaces
        i = j = 0
        while i < len(words):
            word = words[i]
            phone = phones[j]
            if word.mark is not None and word.mark != '':
                i += 1
                j += len(dictionary[word.mark])
                continue
            if word.maxTime - word.minTime >= min_space:
                word.mark = 'SP'
                phone.mark = 'SP'
               
            else:
                word.mark = 'AP'
                phone.mark = 'AP'
            i += 1
            j += 1
            continue
        textgrid.write(str(dst / tgfile.name))



import numpy as np
from paddlespeech.cli.asr.infer import ASRExecutor

def asr_data(audio):
    asr = ASRExecutor()
    result = asr(audio_file=audio.name,force_yes=True)
    print(result)

    return result

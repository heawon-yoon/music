from pydub import AudioSegment

def merge_wav(voice,rhythm):
    # 加载人声音频和节奏音频
    voice_audio  = AudioSegment.from_mp3(voice)
    
    rhythm_audio = AudioSegment.from_file(rhythm, format="wav")
    
    # 按照较长的音频进行对齐
    if len(voice_audio) > len(rhythm_audio):
        rhythm_audio = rhythm_audio[:len(voice_audio)]
    else:
        voice_audio = voice_audio[:len(rhythm_audio)]
    voice_audio = voice_audio+16
    merged_audio = voice_audio.overlay(rhythm_audio)
    merged_audio.export("merged.wav", format="wav")
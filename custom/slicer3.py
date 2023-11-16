import numpy as np
import librosa

# This function is obtained from librosa.
def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
    ):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            # 计算开始时间
            start_time = librosa.frames_to_time(begin,sr=44100)
            print(f'Start time for target shape : {start_time} seconds')
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    # @timeit
    def slice(self, waveform1,sr):
        waveform = librosa.resample(waveform1, sr, 44100, fix=True, scale=False) 

        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        target_sr = 44100
        rms_list = librosa.feature.rms(y=waveform).squeeze(0)
        sil_tags = []
        sil_tags_pos = []
        
        silence_start = None
        clip_start = 0
        print(len(rms_list))
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                    sil_tags_pos.append(i)
                else:
                    sil_tags.append((pos, pos))
                    sil_tags_pos.append(i)
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    sil_tags_pos.append(i)
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
                    sil_tags_pos.append(i)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    sil_tags_pos.append(i)
                else:
                    sil_tags.append((pos_l, pos_r))
                    sil_tags_pos.append(i)
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        print(f"total_frames : {total_frames}")
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
            sil_tags_pos.append(total_frames + 1)
        print(f"sil_tags : {sil_tags}")
        print(f"sil_tags_pos : {sil_tags_pos}")
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            return chunks

def predata(wav_path,out=None,db_thresh=-40,min_length=5000,min_interval=300,hop_size=10,max_sil_kept=5000):
    import os.path
    import librosa
    import soundfile


    if out is None:
        out = os.path.dirname(os.path.abspath(wav_path))+"/slice"
    audio, sr = librosa.load(wav_path, sr=44100, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=db_thresh,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept,
    )
    chunks = slicer.slice(audio,sr)
    rst_datas = [];
    rst_data = {}
    if not os.path.exists(out):
        os.makedirs(out)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(
            os.path.join(
                out,
                f"%s_%d.wav"
                % (os.path.basename(wav_path).rsplit(".", maxsplit=1)[0], i),
                ),
            chunk,
            sr,
        )
        rst_data = {}
        rst_data["wav_path"]= os.path.join(
            out,
            f"%s_%d.wav"
            % (os.path.basename(wav_path).rsplit(".", maxsplit=1)[0], i),
            )
        rst_datas.append(rst_data)
        print(f"rst_data_wav : {rst_data}")
    print(f"rst_datas : {rst_datas}")
    return rst_datas

def main():
    import os.path
    from argparse import ArgumentParser

    import librosa
    import soundfile

    parser = ArgumentParser()
    parser.add_argument("audio", type=str, help="The audio to be sliced")
    parser.add_argument(
        "--out", type=str, help="Output directory of the sliced audio clips"
    )
    parser.add_argument(
        "--db_thresh",
        type=float,
        required=False,
        default=-40,
        help="The dB threshold for silence detection",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        required=False,
        default=5000,
        help="The minimum milliseconds required for each sliced audio clip",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        required=False,
        default=300,
        help="The minimum milliseconds for a silence part to be sliced",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        required=False,
        default=10,
        help="Frame length in milliseconds",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        required=False,
        default=500,
        help="The maximum silence length kept around the sliced clip, presented in milliseconds",
    )
    #args = parser.parse_args()
    parser.parse_known_args()[0]
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    audio, sr = librosa.load(args.audio, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
    )
    chunks = slicer.slice(audio)
    if not os.path.exists(out):
        os.makedirs(out)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(
            os.path.join(
                out,
                f"%s_%d.wav"
                % (os.path.basename(args.audio).rsplit(".", maxsplit=1)[0], i),
            ),
            chunk,
            sr,
        )


if __name__ == "__main__":
    main()

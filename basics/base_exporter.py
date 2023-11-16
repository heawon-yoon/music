import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from utils.hparams import hparams


class BaseExporter:
    def __init__(
            self,
            device: Union[str, torch.device] = None,
            cache_dir: Path = None,
            **kwargs
    ):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir: Path = cache_dir.resolve() if cache_dir is not None \
            else Path(__file__).parent.parent / 'deployment' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # noinspection PyMethodMayBeStatic
    def build_spk_map(self) -> dict:
        if hparams['use_spk_id']:
            with open(Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                spk_map = json.load(f)
            assert isinstance(spk_map, dict) and len(spk_map) > 0, 'Invalid or empty speaker map!'
            assert len(spk_map) == len(set(spk_map.values())), 'Duplicate speaker id in speaker map!'
            return spk_map
        else:
            return {}

    def build_model(self) -> nn.Module:
        """
        Creates an instance of nn.Module and load its state dict on the target device.
        """
        raise NotImplementedError()

    def export_model(self, path: Path):
        """
        Exports the model to ONNX format.
        :param path: the target model path
        """
        raise NotImplementedError()

    def export_attachments(self, path: Path):
        """
        Exports related files and configs (e.g. the dictionary) to the target directory.
        :param path: the target directory
        """
        raise NotImplementedError()

    def export(self, path: Path):
        """
        Exports all the artifacts to the target directory.
        :param path: the target directory
        """
        raise NotImplementedError()

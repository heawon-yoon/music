class BaseVocoder:
    def to_device(self, device):
        """
        
        :param device: torch.device or str
        """
        raise NotImplementedError()

    def get_device(self):
        """
        
        :return: device: torch.device or str
        """
        raise NotImplementedError()
    
    def spec2wav(self, mel, **kwargs):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError()

    @staticmethod
    def wav2spec(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError()

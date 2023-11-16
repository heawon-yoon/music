import shutil
from pathlib import Path
from typing import Union, List, Tuple, Dict

import onnx
import onnxsim
import torch

from basics.base_exporter import BaseExporter
from deployment.modules.toplevel import DiffSingerVarianceONNX
from utils import load_ckpt, onnx_helper, remove_suffix
from utils.hparams import hparams
from utils.phoneme_utils import locate_dictionary, build_phoneme_list
from utils.text_encoder import TokenTextEncoder


class DiffSingerVarianceExporter(BaseExporter):
    def __init__(
            self,
            device: Union[str, torch.device] = 'cpu',
            cache_dir: Path = None,
            ckpt_steps: int = None,
            expose_expr: bool = False,
            export_spk: List[Tuple[str, Dict[str, float]]] = None,
            freeze_spk: Tuple[str, Dict[str, float]] = None
    ):
        super().__init__(device=device, cache_dir=cache_dir)
        # Basic attributes
        self.model_name: str = hparams['exp_name']
        self.ckpt_steps: int = ckpt_steps
        self.spk_map: dict = self.build_spk_map()
        self.vocab = TokenTextEncoder(vocab_list=build_phoneme_list())
        self.model = self.build_model()
        self.linguistic_encoder_cache_path = self.cache_dir / 'linguistic.onnx'
        self.dur_predictor_cache_path = self.cache_dir / 'dur.onnx'
        self.pitch_preprocess_cache_path = self.cache_dir / 'pitch_pre.onnx'
        self.pitch_diffusion_cache_path = self.cache_dir / 'pitch.onnx'
        self.pitch_postprocess_cache_path = self.cache_dir / 'pitch_post.onnx'
        self.variance_preprocess_cache_path = self.cache_dir / 'variance_pre.onnx'
        self.variance_diffusion_cache_path = self.cache_dir / 'variance.onnx'
        self.variance_postprocess_cache_path = self.cache_dir / 'variance_post.onnx'

        # Attributes for logging
        self.fs2_class_name = remove_suffix(self.model.fs2.__class__.__name__, 'ONNX')
        self.dur_predictor_class_name = \
            remove_suffix(self.model.fs2.dur_predictor.__class__.__name__, 'ONNX') \
            if self.model.predict_dur else None
        self.pitch_denoiser_class_name = \
            remove_suffix(self.model.pitch_predictor.denoise_fn.__class__.__name__, 'ONNX') \
            if self.model.predict_pitch else None
        self.pitch_diffusion_class_name = \
            remove_suffix(self.model.pitch_predictor.__class__.__name__, 'ONNX') \
            if self.model.predict_pitch else None
        self.variance_denoiser_class_name = \
            remove_suffix(self.model.variance_predictor.denoise_fn.__class__.__name__, 'ONNX') \
            if self.model.predict_variances else None
        self.variance_diffusion_class_name = \
            remove_suffix(self.model.variance_predictor.__class__.__name__, 'ONNX') \
            if self.model.predict_variances else None

        # Attributes for exporting
        self.expose_expr = expose_expr
        self.freeze_spk: Tuple[str, Dict[str, float]] = freeze_spk \
            if hparams['use_spk_id'] else None
        self.export_spk: List[Tuple[str, Dict[str, float]]] = export_spk \
            if hparams['use_spk_id'] and export_spk is not None else []
        if hparams['use_spk_id']:
            if not self.export_spk and self.freeze_spk is None:
                # In case the user did not specify any speaker settings:
                if len(self.spk_map) == 1:
                    # If there is only one speaker, freeze him/her.
                    first_spk = next(iter(self.spk_map.keys()))
                    self.freeze_spk = (first_spk, {first_spk: 1.0})
                else:
                    # If there are multiple speakers, export them all.
                    self.export_spk = [(name, {name: 1.0}) for name in self.spk_map.keys()]
            if self.freeze_spk is not None:
                self.model.register_buffer('frozen_spk_embed', self._perform_spk_mix(self.freeze_spk[1]))

    def build_model(self) -> DiffSingerVarianceONNX:
        model = DiffSingerVarianceONNX(
            vocab_size=len(self.vocab)
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=self.ckpt_steps,
                  prefix_in_ckpt='model', strict=True, device=self.device)
        model.build_smooth_op(self.device)
        return model

    def export(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        model_name = self.model_name
        if self.freeze_spk is not None:
            model_name += '.' + self.freeze_spk[0]
        self.export_model(path, model_name)
        self.export_attachments(path)

    def export_model(self, path: Path, model_name: str = None):
        self._torch_export_model()
        linguistic_onnx = self._optimize_linguistic_graph(onnx.load(self.linguistic_encoder_cache_path))
        linguistic_path = path / f'{model_name}.linguistic.onnx'
        onnx.save(linguistic_onnx, linguistic_path)
        print(f'| export linguistic encoder => {linguistic_path}')
        self.linguistic_encoder_cache_path.unlink()
        if self.model.predict_dur:
            dur_predictor_onnx = self._optimize_dur_predictor_graph(onnx.load(self.dur_predictor_cache_path))
            dur_predictor_path = path / f'{model_name}.dur.onnx'
            onnx.save(dur_predictor_onnx, dur_predictor_path)
            self.dur_predictor_cache_path.unlink()
            print(f'| export dur predictor => {dur_predictor_path}')
        if self.model.predict_pitch:
            pitch_predictor_onnx = self._optimize_merge_pitch_predictor_graph(
                onnx.load(self.pitch_preprocess_cache_path),
                onnx.load(self.pitch_diffusion_cache_path),
                onnx.load(self.pitch_postprocess_cache_path)
            )
            pitch_predictor_path = path / f'{model_name}.pitch.onnx'
            onnx.save(pitch_predictor_onnx, pitch_predictor_path)
            self.pitch_preprocess_cache_path.unlink()
            self.pitch_diffusion_cache_path.unlink()
            self.pitch_postprocess_cache_path.unlink()
            print(f'| export pitch predictor => {pitch_predictor_path}')
        if self.model.predict_variances:
            variance_predictor_onnx = self._optimize_merge_variance_predictor_graph(
                onnx.load(self.variance_preprocess_cache_path),
                onnx.load(self.variance_diffusion_cache_path),
                onnx.load(self.variance_postprocess_cache_path)
            )
            variance_predictor_path = path / f'{model_name}.variance.onnx'
            onnx.save(variance_predictor_onnx, variance_predictor_path)
            self.variance_preprocess_cache_path.unlink()
            self.variance_diffusion_cache_path.unlink()
            self.variance_postprocess_cache_path.unlink()
            print(f'| export variance predictor => {variance_predictor_path}')

    def export_attachments(self, path: Path):
        for spk in self.export_spk:
            self._export_spk_embed(
                path / f'{self.model_name}.{spk[0]}.emb',
                self._perform_spk_mix(spk[1])
            )
        self._export_dictionary(path / 'dictionary.txt')
        self._export_phonemes((path / f'{self.model_name}.phonemes.txt'))

    @torch.no_grad()
    def _torch_export_model(self):
        # Prepare inputs for FastSpeech2 and dur predictor tracing
        tokens = torch.LongTensor([[1] * 5]).to(self.device)
        ph_dur = torch.LongTensor([[3, 5, 2, 1, 4]]).to(self.device)
        word_div = torch.LongTensor([[2, 2, 1]]).to(self.device)
        word_dur = torch.LongTensor([[8, 3, 4]]).to(self.device)
        encoder_out = torch.rand(1, 5, hparams['hidden_size'], dtype=torch.float32, device=self.device)
        x_masks = tokens == 0
        ph_midi = torch.LongTensor([[60] * 5]).to(self.device)
        encoder_output_names = ['encoder_out', 'x_masks']
        encoder_common_axes = {
            'encoder_out': {
                1: 'n_tokens'
            },
            'x_masks': {
                1: 'n_tokens'
            }
        }
        input_spk_embed = hparams['use_spk_id'] and not self.freeze_spk

        print(f'Exporting {self.fs2_class_name}...')
        if self.model.predict_dur:
            torch.onnx.export(
                self.model.view_as_linguistic_encoder(),
                (
                    tokens,
                    word_div,
                    word_dur
                ),
                self.linguistic_encoder_cache_path,
                input_names=[
                    'tokens',
                    'word_div',
                    'word_dur'
                ],
                output_names=encoder_output_names,
                dynamic_axes={
                    'tokens': {
                        1: 'n_tokens'
                    },
                    'word_div': {
                        1: 'n_words'
                    },
                    'word_dur': {
                        1: 'n_words'
                    },
                    **encoder_common_axes
                },
                opset_version=15
            )

            print(f'Exporting {self.dur_predictor_class_name}...')
            torch.onnx.export(
                self.model.view_as_dur_predictor(),
                (
                    encoder_out,
                    x_masks,
                    ph_midi,
                    *([torch.rand(
                        1, 5, hparams['hidden_size'],
                        dtype=torch.float32, device=self.device
                    )] if input_spk_embed else [])
                ),
                self.dur_predictor_cache_path,
                input_names=[
                    'encoder_out',
                    'x_masks',
                    'ph_midi',
                    *(['spk_embed'] if input_spk_embed else [])
                ],
                output_names=[
                    'ph_dur_pred'
                ],
                dynamic_axes={
                    'ph_midi': {
                        1: 'n_tokens'
                    },
                    'ph_dur_pred': {
                        1: 'n_tokens'
                    },
                    **({'spk_embed': {1: 'n_tokens'}} if input_spk_embed else {}),
                    **encoder_common_axes
                },
                opset_version=15
            )
        else:
            torch.onnx.export(
                self.model.view_as_linguistic_encoder(),
                (
                    tokens,
                    ph_dur
                ),
                self.linguistic_encoder_cache_path,
                input_names=[
                    'tokens',
                    'ph_dur'
                ],
                output_names=encoder_output_names,
                dynamic_axes={
                    'tokens': {
                        1: 'n_tokens'
                    },
                    'ph_dur': {
                        1: 'n_tokens'
                    },
                    **encoder_common_axes
                },
                opset_version=15
            )

        if self.model.predict_pitch:
            use_melody_encoder = hparams.get('use_melody_encoder', False)
            use_glide_embed = use_melody_encoder and hparams['use_glide_embed']
            # Prepare inputs for preprocessor of PitchDiffusion
            note_midi = torch.FloatTensor([[60.] * 4]).to(self.device)
            note_dur = torch.LongTensor([[2, 6, 3, 4]]).to(self.device)
            pitch = torch.FloatTensor([[60.] * 15]).to(self.device)
            retake = torch.ones_like(pitch, dtype=torch.bool)
            pitch_input_args = (
                encoder_out,
                ph_dur,
                {
                    'note_midi': note_midi,
                    **({'note_rest': note_midi >= 0} if use_melody_encoder else {}),
                    'note_dur': note_dur,
                    **({'note_glide': torch.zeros_like(note_midi, dtype=torch.long)} if use_glide_embed else {}),
                    'pitch': pitch,
                    **({'expr': torch.ones_like(pitch)} if self.expose_expr else {}),
                    'retake': retake,
                    **({'spk_embed': torch.rand(
                        1, 15, hparams['hidden_size'], dtype=torch.float32, device=self.device
                    )} if input_spk_embed else {})
                }
            )
            torch.onnx.export(
                self.model.view_as_pitch_preprocess(),
                pitch_input_args,
                self.pitch_preprocess_cache_path,
                input_names=[
                    'encoder_out', 'ph_dur', 'note_midi',
                    *(['note_rest'] if use_melody_encoder else []),
                    'note_dur',
                    *(['note_glide'] if use_glide_embed else []),
                    'pitch',
                    *(['expr'] if self.expose_expr else []),
                    'retake',
                    *(['spk_embed'] if input_spk_embed else [])
                ],
                output_names=[
                    'pitch_cond', 'base_pitch'
                ],
                dynamic_axes={
                    'encoder_out': {
                        1: 'n_tokens'
                    },
                    'ph_dur': {
                        1: 'n_tokens'
                    },
                    'note_midi': {
                        1: 'n_notes'
                    },
                    **({'note_rest': {1: 'n_notes'}} if use_melody_encoder else {}),
                    'note_dur': {
                        1: 'n_notes'
                    },
                    **({'note_glide': {1: 'n_notes'}} if use_glide_embed else {}),
                    'pitch': {
                        1: 'n_frames'
                    },
                    **({'expr': {1: 'n_frames'}} if self.expose_expr else {}),
                    'retake': {
                        1: 'n_frames'
                    },
                    'pitch_cond': {
                        1: 'n_frames'
                    },
                    'base_pitch': {
                        1: 'n_frames'
                    },
                    **({'spk_embed': {1: 'n_frames'}} if input_spk_embed else {})
                },
                opset_version=15
            )

            # Prepare inputs for denoiser tracing and PitchDiffusion scripting
            shape = (1, 1, hparams['pitch_prediction_args']['repeat_bins'], 15)
            noise = torch.randn(shape, device=self.device)
            condition = torch.rand((1, hparams['hidden_size'], 15), device=self.device)
            step = (torch.rand((1,), device=self.device) * hparams['K_step']).long()

            print(f'Tracing {self.pitch_denoiser_class_name} denoiser...')
            pitch_diffusion = self.model.view_as_pitch_diffusion()
            pitch_diffusion.pitch_predictor.denoise_fn = torch.jit.trace(
                pitch_diffusion.pitch_predictor.denoise_fn,
                (
                    noise,
                    step,
                    condition
                )
            )

            print(f'Scripting {self.pitch_diffusion_class_name}...')
            pitch_diffusion = torch.jit.script(
                pitch_diffusion,
                example_inputs=[
                    (
                        condition.transpose(1, 2),
                        1  # p_sample branch
                    ),
                    (
                        condition.transpose(1, 2),
                        200  # p_sample_plms branch
                    )
                ]
            )

            print(f'Exporting {self.pitch_diffusion_class_name}...')
            torch.onnx.export(
                pitch_diffusion,
                (
                    condition.transpose(1, 2),
                    200
                ),
                self.pitch_diffusion_cache_path,
                input_names=[
                    'pitch_cond', 'speedup'
                ],
                output_names=[
                    'x_pred'
                ],
                dynamic_axes={
                    'pitch_cond': {
                        1: 'n_frames'
                    },
                    'x_pred': {
                        1: 'n_frames'
                    }
                },
                opset_version=15
            )

            # Prepare inputs for postprocessor of MultiVarianceDiffusion
            torch.onnx.export(
                self.model.view_as_pitch_postprocess(),
                (
                    pitch,
                    pitch
                ),
                self.pitch_postprocess_cache_path,
                input_names=[
                    'x_pred',
                    'base_pitch'
                ],
                output_names=[
                    'pitch_pred'
                ],
                dynamic_axes={
                    'x_pred': {
                        1: 'n_frames'
                    },
                    'base_pitch': {
                        1: 'n_frames'
                    },
                    'pitch_pred': {
                        1: 'n_frames'
                    }
                },
                opset_version=15
            )

        if self.model.predict_variances:
            total_repeat_bins = hparams['variances_prediction_args']['total_repeat_bins']
            repeat_bins = total_repeat_bins // len(self.model.variance_prediction_list)

            # Prepare inputs for preprocessor of MultiVarianceDiffusion
            pitch = torch.FloatTensor([[60.] * 15]).to(self.device)
            variances = {
                v_name: torch.FloatTensor([[0.] * 15]).to(self.device)
                for v_name in self.model.variance_prediction_list
            }
            retake = torch.ones_like(pitch, dtype=torch.bool)[..., None].tile(len(self.model.variance_prediction_list))
            torch.onnx.export(
                self.model.view_as_variance_preprocess(),
                (
                    encoder_out,
                    ph_dur,
                    pitch,
                    variances,
                    retake,
                    *([torch.rand(
                        1, 15, hparams['hidden_size'],
                        dtype=torch.float32, device=self.device
                    )] if input_spk_embed else [])
                ),
                self.variance_preprocess_cache_path,
                input_names=[
                    'encoder_out', 'ph_dur', 'pitch',
                    *self.model.variance_prediction_list,
                    'retake',
                    *(['spk_embed'] if input_spk_embed else [])
                ],
                output_names=[
                    'variance_cond'
                ],
                dynamic_axes={
                    'encoder_out': {
                        1: 'n_tokens'
                    },
                    'ph_dur': {
                        1: 'n_tokens'
                    },
                    'pitch': {
                        1: 'n_frames'
                    },
                    **{
                        v_name: {
                            1: 'n_frames'
                        }
                        for v_name in self.model.variance_prediction_list
                    },
                    'retake': {
                        1: 'n_frames'
                    },
                    **({'spk_embed': {1: 'n_frames'}} if input_spk_embed else {})
                },
                opset_version=15
            )

            # Prepare inputs for denoiser tracing and MultiVarianceDiffusion scripting
            shape = (1, len(self.model.variance_prediction_list), repeat_bins, 15)
            noise = torch.randn(shape, device=self.device)
            condition = torch.rand((1, hparams['hidden_size'], 15), device=self.device)
            step = (torch.rand((1,), device=self.device) * hparams['K_step']).long()

            print(f'Tracing {self.variance_denoiser_class_name} denoiser...')
            variance_diffusion = self.model.view_as_variance_diffusion()
            variance_diffusion.variance_predictor.denoise_fn = torch.jit.trace(
                variance_diffusion.variance_predictor.denoise_fn,
                (
                    noise,
                    step,
                    condition
                )
            )

            print(f'Scripting {self.variance_diffusion_class_name}...')
            variance_diffusion = torch.jit.script(
                variance_diffusion,
                example_inputs=[
                    (
                        condition.transpose(1, 2),
                        1  # p_sample branch
                    ),
                    (
                        condition.transpose(1, 2),
                        200  # p_sample_plms branch
                    )
                ]
            )

            print(f'Exporting {self.variance_diffusion_class_name}...')
            torch.onnx.export(
                variance_diffusion,
                (
                    condition.transpose(1, 2),
                    200
                ),
                self.variance_diffusion_cache_path,
                input_names=[
                    'variance_cond', 'speedup'
                ],
                output_names=[
                    'xs_pred'
                ],
                dynamic_axes={
                    'variance_cond': {
                        1: 'n_frames'
                    },
                    'xs_pred': {
                        (1 if len(self.model.variance_prediction_list) == 1 else 2): 'n_frames'
                    }
                },
                opset_version=15
            )

            # Prepare inputs for postprocessor of MultiVarianceDiffusion
            xs_shape = (1, 15) \
                if len(self.model.variance_prediction_list) == 1 \
                else (1, len(self.model.variance_prediction_list), 15)
            xs_pred = torch.randn(xs_shape, dtype=torch.float32, device=self.device)
            torch.onnx.export(
                self.model.view_as_variance_postprocess(),
                (
                    xs_pred
                ),
                self.variance_postprocess_cache_path,
                input_names=[
                    'xs_pred'
                ],
                output_names=[
                    f'{v_name}_pred'
                    for v_name in self.model.variance_prediction_list
                ],
                dynamic_axes={
                    'xs_pred': {
                        (1 if len(self.model.variance_prediction_list) == 1 else 2): 'n_frames'
                    },
                    **{
                        f'{v_name}_pred': {
                            1: 'n_frames'
                        }
                        for v_name in self.model.variance_prediction_list
                    }
                },
                opset_version=15
            )

    @torch.no_grad()
    def _perform_spk_mix(self, spk_mix: Dict[str, float]):
        spk_mix_ids = []
        spk_mix_values = []
        for name, value in spk_mix.items():
            spk_mix_ids.append(self.spk_map[name])
            assert value >= 0., f'Speaker mix checks failed.\n' \
                                f'Proportion of speaker \'{name}\' is negative.'
            spk_mix_values.append(value)
        spk_mix_id_N = torch.LongTensor(spk_mix_ids).to(self.device)[None]  # => [1, N]
        spk_mix_value_N = torch.FloatTensor(spk_mix_values).to(self.device)[None]  # => [1, N]
        spk_mix_value_sum = spk_mix_value_N.sum()
        assert spk_mix_value_sum > 0., f'Speaker mix checks failed.\n' \
                                       f'Proportions of speaker mix sum to zero.'
        spk_mix_value_N /= spk_mix_value_sum  # normalize
        spk_mix_embed = torch.sum(
            self.model.spk_embed(spk_mix_id_N) * spk_mix_value_N.unsqueeze(2),  # => [1, N, H]
            dim=1, keepdim=False
        )  # => [1, H]
        return spk_mix_embed

    def _optimize_linguistic_graph(self, linguistic: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_override_io_shapes(
            linguistic,
            output_shapes={
                'encoder_out': (1, 'n_tokens', hparams['hidden_size'])
            }
        )
        print(f'Running ONNX Simplifier on {self.fs2_class_name}...')
        linguistic, check = onnxsim.simplify(linguistic, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        print(f'| optimize graph: {self.fs2_class_name}')
        return linguistic

    def _optimize_dur_predictor_graph(self, dur_predictor: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_override_io_shapes(
            dur_predictor,
            output_shapes={
                'ph_dur_pred': (1, 'n_tokens')
            }
        )
        print(f'Running ONNX Simplifier on {self.dur_predictor_class_name}...')
        dur_predictor, check = onnxsim.simplify(dur_predictor, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        print(f'| optimize graph: {self.dur_predictor_class_name}')
        return dur_predictor

    def _optimize_merge_pitch_predictor_graph(
            self, pitch_pre: onnx.ModelProto, pitch_diffusion: onnx.ModelProto, pitch_post: onnx.ModelProto
    ) -> onnx.ModelProto:
        onnx_helper.model_override_io_shapes(
            pitch_pre, output_shapes={'pitch_cond': (1, 'n_frames', hparams['hidden_size'])}
        )
        pitch_pre, check = onnxsim.simplify(pitch_pre, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'

        onnx_helper.model_override_io_shapes(
            pitch_diffusion, output_shapes={'pitch_pred': (1, 'n_frames')}
        )
        print(f'Running ONNX Simplifier #1 on {self.pitch_diffusion_class_name}...')
        pitch_diffusion, check = onnxsim.simplify(pitch_diffusion, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx_helper.graph_fold_back_to_squeeze(pitch_diffusion.graph)
        onnx_helper.graph_extract_conditioner_projections(
            graph=pitch_diffusion.graph, op_type='Conv',
            weight_pattern=r'pitch_predictor\.denoise_fn\.residual_layers\.\d+\.conditioner_projection\.weight',
            alias_prefix='/pitch_predictor/denoise_fn/cache'
        )
        onnx_helper.graph_remove_unused_values(pitch_diffusion.graph)
        print(f'Running ONNX Simplifier #2 on {self.pitch_diffusion_class_name}...')
        pitch_diffusion, check = onnxsim.simplify(pitch_diffusion, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'

        onnx_helper.model_add_prefixes(pitch_pre, node_prefix='/pre', ignored_pattern=r'.*embed.*')
        onnx_helper.model_add_prefixes(pitch_pre, dim_prefix='pre.', ignored_pattern='(n_tokens)|(n_notes)|(n_frames)')
        onnx_helper.model_add_prefixes(pitch_post, node_prefix='/post', ignored_pattern=None)
        onnx_helper.model_add_prefixes(pitch_post, dim_prefix='post.', ignored_pattern='n_frames')
        pitch_pre_diffusion = onnx.compose.merge_models(
            pitch_pre, pitch_diffusion, io_map=[('pitch_cond', 'pitch_cond')],
            prefix1='', prefix2='', doc_string='',
            producer_name=pitch_pre.producer_name, producer_version=pitch_pre.producer_version,
            domain=pitch_pre.domain, model_version=pitch_pre.model_version
        )
        pitch_pre_diffusion.graph.name = pitch_pre.graph.name
        pitch_predictor = onnx.compose.merge_models(
            pitch_pre_diffusion, pitch_post, io_map=[
                ('x_pred', 'x_pred'), ('base_pitch', 'base_pitch')
            ], prefix1='', prefix2='', doc_string='',
            producer_name=pitch_pre.producer_name, producer_version=pitch_pre.producer_version,
            domain=pitch_pre.domain, model_version=pitch_pre.model_version
        )
        pitch_predictor.graph.name = pitch_pre.graph.name

        print(f'| optimize graph: {self.pitch_diffusion_class_name}')
        return pitch_predictor

    def _optimize_merge_variance_predictor_graph(
            self, var_pre: onnx.ModelProto, var_diffusion: onnx.ModelProto, var_post: onnx.ModelProto
    ):
        onnx_helper.model_override_io_shapes(
            var_pre, output_shapes={'variance_cond': (1, 'n_frames', hparams['hidden_size'])}
        )
        var_pre, check = onnxsim.simplify(var_pre, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'

        onnx_helper.model_override_io_shapes(
            var_diffusion, output_shapes={
                'xs_pred': (1, 'n_frames')
                if len(self.model.variance_prediction_list) == 1
                else (1, len(self.model.variance_prediction_list), 'n_frames')
            }
        )
        print(f'Running ONNX Simplifier #1 on'
              f' {self.variance_diffusion_class_name}...')
        var_diffusion, check = onnxsim.simplify(var_diffusion, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx_helper.graph_fold_back_to_squeeze(var_diffusion.graph)
        onnx_helper.graph_extract_conditioner_projections(
            graph=var_diffusion.graph, op_type='Conv',
            weight_pattern=r'variance_predictor\.denoise_fn\.residual_layers\.\d+\.conditioner_projection\.weight',
            alias_prefix='/variance_predictor/denoise_fn/cache'
        )
        onnx_helper.graph_remove_unused_values(var_diffusion.graph)
        print(f'Running ONNX Simplifier #2 on {self.variance_diffusion_class_name}...')
        var_diffusion, check = onnxsim.simplify(var_diffusion, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'

        var_post, check = onnxsim.simplify(var_post, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'

        ignored_variance_names = '|'.join([f'({v_name})' for v_name in self.model.variance_prediction_list])
        onnx_helper.model_add_prefixes(
            var_pre, node_prefix='/pre', value_info_prefix='/pre',
            ignored_pattern=fr'.*((embed)|{ignored_variance_names}).*'
        )
        onnx_helper.model_add_prefixes(var_pre, dim_prefix='pre.', ignored_pattern='(n_tokens)|(n_frames)')
        onnx_helper.model_add_prefixes(
            var_post, node_prefix='/post', value_info_prefix='/post',
            ignored_pattern=None
        )
        onnx_helper.model_add_prefixes(var_post, dim_prefix='post.', ignored_pattern='n_frames')

        print(f'Merging {self.variance_diffusion_class_name} subroutines...')
        var_pre_diffusion = onnx.compose.merge_models(
            var_pre, var_diffusion, io_map=[('variance_cond', 'variance_cond')],
            prefix1='', prefix2='', doc_string='',
            producer_name=var_pre.producer_name, producer_version=var_pre.producer_version,
            domain=var_pre.domain, model_version=var_pre.model_version
        )
        var_pre_diffusion.graph.name = var_pre.graph.name
        var_predictor = onnx.compose.merge_models(
            var_pre_diffusion, var_post, io_map=[('xs_pred', 'xs_pred')],
            prefix1='', prefix2='', doc_string='',
            producer_name=var_pre.producer_name, producer_version=var_pre.producer_version,
            domain=var_pre.domain, model_version=var_pre.model_version
        )
        var_predictor.graph.name = var_pre.graph.name
        return var_predictor

    # noinspection PyMethodMayBeStatic
    def _export_spk_embed(self, path: Path, spk_embed: torch.Tensor):
        with open(path, 'wb') as f:
            f.write(spk_embed.cpu().numpy().tobytes())
        print(f'| export spk embed => {path}')

    # noinspection PyMethodMayBeStatic
    def _export_dictionary(self, path: Path):
        print(f'| export dictionary => {path}')
        shutil.copy(locate_dictionary(), path)

    def _export_phonemes(self, path: Path):
        self.vocab.store_to_file(path)
        print(f'| export phonemes => {path}')

import logging
import os
import pathlib
import shutil
import sys
from datetime import datetime
from typing import Dict

import matplotlib

import utils
from utils.text_encoder import TokenTextEncoder

matplotlib.use('Agg')

import torch.utils.data
from torchmetrics import Metric, MeanMetric
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only

from basics.base_module import CategorizedModule
from utils.hparams import hparams
from utils.training_utils import (
    DsModelCheckpoint, DsTQDMProgressBar,
    DsBatchSampler, DsEvalBatchSampler,
    get_latest_checkpoint_path, get_strategy
)
from utils.phoneme_utils import locate_dictionary, build_phoneme_list

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseTask(pl.LightningModule):
    """
        Base class for training tasks.
        1. *load_ckpt*:
            load checkpoint;
        2. *training_step*:
            record and log the loss;
        3. *optimizer_step*:
            run backwards step;
        4. *start*:
            load training configs, backup code, log to tensorboard, start training;
        5. *configure_ddp* and *init_ddp_connection*:
            start parallel training.

        Subclasses should define:
        1. *build_model*, *build_optimizer*, *build_scheduler*:
            how to build the model, the optimizer and the training scheduler;
        2. *_training_step*:
            one training step of the model;
        3. *on_validation_end* and *_on_validation_end*:
            postprocess the validation output.
    """

    def __init__(self, *args, **kwargs):
        # dataset configs
        super().__init__(*args, **kwargs)
        self.dataset_cls = None
        self.max_batch_frames = hparams['max_batch_frames']
        self.max_batch_size = hparams['max_batch_size']
        self.max_val_batch_frames = hparams['max_val_batch_frames']
        if self.max_val_batch_frames == -1:
            hparams['max_val_batch_frames'] = self.max_val_batch_frames = self.max_batch_frames
        self.max_val_batch_size = hparams['max_val_batch_size']
        if self.max_val_batch_size == -1:
            hparams['max_val_batch_size'] = self.max_val_batch_size = self.max_batch_size

        self.training_sampler = None
        self.model = None
        self.skip_immediate_validation = False
        self.skip_immediate_ckpt_save = False

        self.valid_losses: Dict[str, Metric] = {
            'total_loss': MeanMetric()
        }
        self.valid_metric_names = set()

    ###########
    # Training, validation and testing
    ###########
    def setup(self, stage):
        self.phone_encoder = self.build_phone_encoder()
        self.model = self.build_model()
        # utils.load_warp(self)
        self.unfreeze_all_params()
        if hparams['freezing_enabled']:
            self.freeze_params()
        if hparams['finetune_enabled'] and get_latest_checkpoint_path(pathlib.Path(hparams['work_dir'])) is None:
            self.load_finetune_ckpt(self.load_pre_train_model())
        self.print_arch()
        self.build_losses_and_metrics()
        self.train_dataset = self.dataset_cls(hparams['train_set_name'])
        self.valid_dataset = self.dataset_cls(hparams['valid_set_name'])

    def get_need_freeze_state_dict_key(self, model_state_dict) -> list:
        key_list = []
        for i in hparams['frozen_params']:
            for j in model_state_dict:
                if j.startswith(i):
                    key_list.append(j)
        return list(set(key_list))

    def freeze_params(self) -> None:
        model_state_dict = self.state_dict().keys()
        freeze_key = self.get_need_freeze_state_dict_key(model_state_dict=model_state_dict)

        for i in freeze_key:
            params=self.get_parameter(i)

            params.requires_grad = False

    def unfreeze_all_params(self) -> None:
        for i in self.model.parameters():
            i.requires_grad = True

    def load_finetune_ckpt(
            self, state_dict
    ) -> None:

        adapt_shapes = hparams['finetune_strict_shapes']
        if not adapt_shapes:
            cur_model_state_dict = self.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print('| Unmatched keys: ', key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

    def load_pre_train_model(self):

        pre_train_ckpt_path = hparams.get('finetune_ckpt_path')
        blacklist = hparams.get('finetune_ignored_params')
        # whitelist=hparams.get('pre_train_whitelist')
        if blacklist is None:
            blacklist = []
        # if whitelist is  None:
        #     raise RuntimeError("")

        if pre_train_ckpt_path is not None:
            ckpt = torch.load(pre_train_ckpt_path)
            # if ckpt.get('category') is None:
            #     raise RuntimeError("")

            if isinstance(self.model, CategorizedModule):
                self.model.check_category(ckpt.get('category'))

            state_dict = {}
            for i in ckpt['state_dict']:
                # if 'diffusion' in i:
                # if i in rrrr:
                #     continue
                skip = False
                for b in blacklist:
                    if i.startswith(b):
                        skip = True
                        break

                if skip:
                    continue

                state_dict[i] = ckpt['state_dict'][i]
                print(i)
            return state_dict
        else:
            raise RuntimeError("")

    @staticmethod
    def build_phone_encoder():
        phone_list = build_phoneme_list()
        return TokenTextEncoder(vocab_list=phone_list)

    def build_model(self):
        raise NotImplementedError()

    @rank_zero_only
    def print_arch(self):
        utils.print_arch(self.model)

    def build_losses_and_metrics(self):
        raise NotImplementedError()

    def register_metric(self, name: str, metric: Metric):
        assert isinstance(metric, Metric)
        setattr(self, name, metric)
        self.valid_metric_names.add(name)

    def run_model(self, sample, infer=False):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        raise NotImplementedError()

    def on_train_epoch_start(self):
        if self.training_sampler is not None:
            self.training_sampler.set_epoch(self.current_epoch)

    def _training_step(self, sample):
        """
        :return: total loss: torch.Tensor, loss_log: dict, other_log: dict
        """
        losses = self.run_model(sample)
        total_loss = sum(losses.values())
        return total_loss, {**losses, 'batch_size': float(sample['size'])}

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        total_loss, log_outputs = self._training_step(sample)

        # logs to progress bar
        self.log_dict(log_outputs, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        if self.global_step % hparams['log_interval'] == 0:
            tb_log = {f'training/{k}': v for k, v in log_outputs.items()}
            tb_log['training/lr'] = self.lr_schedulers().get_last_lr()[0]
            self.logger.log_metrics(tb_log, step=self.global_step)

        return total_loss

    # def on_before_optimizer_step(self, *args, **kwargs):
    #     self.log_dict(grad_norm(self, norm_type=2))

    def _on_validation_start(self):
        pass

    def on_validation_start(self):
        self._on_validation_start()
        for metric in self.valid_losses.values():
            metric.to(self.device)
            metric.reset()

    def _validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: loss_log: dict, weight: int
        """
        raise NotImplementedError()

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        """
        if self.skip_immediate_validation:
            rank_zero_debug(f"Skip validation {batch_idx}")
            return {}
        with torch.autocast(self.device.type, enabled=False):
            losses, weight = self._validation_step(sample, batch_idx)
        losses = {
            'total_loss': sum(losses.values()),
            **losses
        }
        for k, v in losses.items():
            if k not in self.valid_losses:
                self.valid_losses[k] = MeanMetric().to(self.device)
            self.valid_losses[k].update(v, weight=weight)
        return losses

    def on_validation_epoch_end(self):
        if self.skip_immediate_validation:
            self.skip_immediate_validation = False
            self.skip_immediate_ckpt_save = True
            return
        loss_vals = {k: v.compute() for k, v in self.valid_losses.items()}
        self.log('val_loss', loss_vals['total_loss'], on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics({f'validation/{k}': v for k, v in loss_vals.items()}, step=self.global_step)
        for metric in self.valid_losses.values():
            metric.reset()
        metric_vals = {k: getattr(self, k).compute() for k in self.valid_metric_names}
        self.logger.log_metrics({f'metrics/{k}': v for k, v in metric_vals.items()}, step=self.global_step)
        for metric_name in self.valid_metric_names:
            getattr(self, metric_name).reset()

    # noinspection PyMethodMayBeStatic
    def build_scheduler(self, optimizer):
        from utils import build_lr_scheduler_from_config

        scheduler_args = hparams['lr_scheduler_args']
        assert scheduler_args['scheduler_cls'] != ''
        scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
        return scheduler

    # noinspection PyMethodMayBeStatic
    def build_optimizer(self, model):
        from utils import build_object_from_class_name

        optimizer_args = hparams['optimizer_args']
        assert optimizer_args['optimizer_cls'] != ''
        if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
            optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])
        optimizer = build_object_from_class_name(
            optimizer_args['optimizer_cls'],
            torch.optim.Optimizer,
            model.parameters(),
            **optimizer_args
        )
        return optimizer

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        scheduler = self.build_scheduler(optm)
        if scheduler is None:
            return optm
        return {
            "optimizer": optm,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        self.training_sampler = DsBatchSampler(
            self.train_dataset,
            max_batch_frames=self.max_batch_frames,
            max_batch_size=self.max_batch_size,
            num_replicas=(self.trainer.distributed_sampler_kwargs or {}).get('num_replicas', 1),
            rank=(self.trainer.distributed_sampler_kwargs or {}).get('rank', 0),
            sort_by_similar_size=hparams['sort_by_len'],
            required_batch_count_multiple=hparams['accumulate_grad_batches'],
            shuffle_sample=True,
            shuffle_batch=False,
            seed=hparams['seed']
        )
        return torch.utils.data.DataLoader(self.train_dataset,
                                           collate_fn=self.train_dataset.collater,
                                           batch_sampler=self.training_sampler,
                                           num_workers=hparams['ds_workers'],
                                           prefetch_factor=hparams['dataloader_prefetch_factor'],
                                           pin_memory=True,
                                           persistent_workers=True)

    def val_dataloader(self):
        sampler = DsEvalBatchSampler(
            self.valid_dataset,
            max_batch_frames=self.max_val_batch_frames,
            max_batch_size=self.max_val_batch_size,
            rank=(self.trainer.distributed_sampler_kwargs or {}).get('rank', 0),
            batch_by_size=False
        )
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           collate_fn=self.valid_dataset.collater,
                                           batch_sampler=sampler,
                                           num_workers=hparams['ds_workers'],
                                           prefetch_factor=hparams['dataloader_prefetch_factor'],
                                           shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

    def on_test_start(self):
        self.on_validation_start()

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def on_test_end(self):
        return self.on_validation_end()

    ###########
    # Running configuration
    ###########

    @classmethod
    def start(cls):
        pl.seed_everything(hparams['seed'], workers=True)
        task = cls()

        # if pre_train is not None:
        #     task.load_state_dict(pre_train,strict=False)
        #     print("load success-------------------------------------------------------------------")

        work_dir = pathlib.Path(hparams['work_dir'])
        trainer = pl.Trainer(
            accelerator=hparams['pl_trainer_accelerator'],
            devices=hparams['pl_trainer_devices'],
            num_nodes=hparams['pl_trainer_num_nodes'],
            strategy=get_strategy(
                accelerator=hparams['pl_trainer_accelerator'],
                devices=hparams['pl_trainer_devices'],
                num_nodes=hparams['pl_trainer_num_nodes'],
                strategy=hparams['pl_trainer_strategy'],
                backend=hparams['ddp_backend']
            ),
            precision=hparams['pl_trainer_precision'],
            callbacks=[
                DsModelCheckpoint(
                    dirpath=work_dir,
                    filename='model_ckpt_steps_{step}',
                    auto_insert_metric_name=False,
                    monitor='step',
                    mode='max',
                    save_last=False,
                    # every_n_train_steps=hparams['val_check_interval'],
                    save_top_k=hparams['num_ckpt_keep'],
                    permanent_ckpt_start=hparams['permanent_ckpt_start'],
                    permanent_ckpt_interval=hparams['permanent_ckpt_interval'],
                    verbose=True
                ),
                # LearningRateMonitor(logging_interval='step'),
                DsTQDMProgressBar(),
            ],
            logger=TensorBoardLogger(
                save_dir=str(work_dir),
                name='lightning_logs',
                version='lastest'
            ),
            gradient_clip_val=hparams['clip_grad_norm'],
            val_check_interval=hparams['val_check_interval'] * hparams['accumulate_grad_batches'],
            # so this is global_steps
            check_val_every_n_epoch=None,
            log_every_n_steps=1,
            max_steps=hparams['max_updates'],
            use_distributed_sampler=False,
            num_sanity_val_steps=hparams['num_sanity_val_steps'],
            accumulate_grad_batches=hparams['accumulate_grad_batches']
        )
        if not hparams['infer']:  # train
            @rank_zero_only
            def train_payload_copy():
                # copy_code = input(f'{hparams["save_codes"]} code backup? y/n: ') == 'y'
                copy_code = True  # backup code every time
                if copy_code:
                    code_dir = work_dir / 'codes' / datetime.now().strftime('%Y%m%d%H%M%S')
                    code_dir.mkdir(exist_ok=True, parents=True)
                    for c in hparams['save_codes']:
                        shutil.copytree(c, code_dir / c, dirs_exist_ok=True)
                    print(f'| Copied codes to {code_dir}.')
                # Copy spk_map.json and dictionary.txt to work dir
                binary_dir = pathlib.Path(hparams['binary_data_dir'])
                spk_map = work_dir / 'spk_map.json'
                spk_map_src = binary_dir / 'spk_map.json'
                if not spk_map.exists() and spk_map_src.exists():
                    shutil.copy(spk_map_src, spk_map)
                    print(f'| Copied spk map to {spk_map}.')
                dictionary = work_dir / 'dictionary.txt'
                dict_src = binary_dir / 'dictionary.txt'
                if not dictionary.exists():
                    if dict_src.exists():
                        shutil.copy(dict_src, dictionary)
                    else:
                        shutil.copy(locate_dictionary(), dictionary)
                    print(f'| Copied dictionary to {dictionary}.')

            train_payload_copy()
            trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))
        else:
            trainer.test(task)

    def on_save_checkpoint(self, checkpoint):
        if isinstance(self.model, CategorizedModule):
            checkpoint['category'] = self.model.category
        checkpoint['trainer_stage'] = self.trainer.state.stage.value

    def on_load_checkpoint(self, checkpoint):
        from lightning.pytorch.trainer.states import RunningStage
        from utils import simulate_lr_scheduler
        if checkpoint.get('trainer_stage', '') == RunningStage.VALIDATING.value:
            self.skip_immediate_validation = True

        optimizer_args = hparams['optimizer_args']
        scheduler_args = hparams['lr_scheduler_args']

        if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
            optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])

        if checkpoint.get('optimizer_states', None):
            opt_states = checkpoint['optimizer_states']
            assert len(opt_states) == 1  # only support one optimizer
            opt_state = opt_states[0]
            for param_group in opt_state['param_groups']:
                for k, v in optimizer_args.items():
                    if k in param_group and param_group[k] != v:
                        if 'lr_schedulers' in checkpoint and checkpoint['lr_schedulers'] and k == 'lr':
                            continue
                        rank_zero_info(f'| Overriding optimizer parameter {k} from checkpoint: {param_group[k]} -> {v}')
                        param_group[k] = v
                if 'initial_lr' in param_group and param_group['initial_lr'] != optimizer_args['lr']:
                    rank_zero_info(
                        f'| Overriding optimizer parameter initial_lr from checkpoint: {param_group["initial_lr"]} -> {optimizer_args["lr"]}'
                    )
                    param_group['initial_lr'] = optimizer_args['lr']

        if checkpoint.get('lr_schedulers', None):
            assert checkpoint.get('optimizer_states', False)
            assert len(checkpoint['lr_schedulers']) == 1  # only support one scheduler
            checkpoint['lr_schedulers'][0] = simulate_lr_scheduler(
                optimizer_args, scheduler_args,
                step_count=checkpoint['global_step'],
                num_param_groups=len(checkpoint['optimizer_states'][0]['param_groups'])
            )
            for param_group, new_lr in zip(
                checkpoint['optimizer_states'][0]['param_groups'],
                checkpoint['lr_schedulers'][0]['_last_lr'],
            ):
                if param_group['lr'] != new_lr:
                    rank_zero_info(f'| Overriding optimizer parameter lr from checkpoint: {param_group["lr"]} -> {new_lr}')
                    param_group['lr'] = new_lr

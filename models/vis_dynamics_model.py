from torchvision.utils import save_image
import os
import pytorch_lightning as pl
from utils.misc import mkdir, tuple2name, mkpath
from utils.show import generate_video_directory
from geomloss import SamplesLoss
from typing import Union
from itertools import chain

from models.latent_autoencoder import *
from models.nsv_autoencoder import *
from models.nsv_mlp import NSVMLP
from models.smooth_nsv_autoencoder import *
from utils.tangent_utils import build_cross_traj_knn_pairs, rbf_pair_weights, rk4_flow_map, secant_transport_loss, transport_secants_jvp


class VisDynamicsModel(pl.LightningModule):

    def __init__(self,
                 model: None,
                 output_dir: str = 'outputs',
                 lr: float = 1e-4,
                 gamma: float = 0.5,
                 lr_schedule: list = [20, 50, 100],
                 reconstruct_loss_type: str = 'high-dim-latent',
                 smooth_loss_type: str = 'none',
                 regularize_loss_type: str = 'none',
                 margin: float = 0.0,
                 reconstruct_loss_weight: float = 1.0,
                 smooth_loss_weight: Union[list, float] = 0.0,
                 regularize_loss_weight: float = 0.0,
                 model_annealing_list: list = [],
                 tangent_step_weight: float = 1.0,
                 tangent_loss_weight: float = 2.0,
                 tangent_norm_weight: float = 1.0,
                 tangent_angle_weight: float = 1.0,
                 tangent_k: int = 8,
                 tangent_warmup_epochs: int = 25,
                 tangent_ramp_epochs: int = 20,
                 tangent_eps: float = 1e-6,
                 **kwargs) -> None:
        super().__init__()

        self.dt = 1/60 if model.dataset != 'cylindrical_flow' else .02
        self.output_dir = output_dir

        self.model = model
        
        self.loss_func = nn.MSELoss(reduction='none')
        self.regularize_loss_func = SamplesLoss(loss='sinkhorn')

        self.lr = lr
        self.gamma = gamma
        self.lr_schedule = lr_schedule

        self.reconstruct_loss_type = reconstruct_loss_type
        self.smooth_loss_type = smooth_loss_type
        self.regularize_loss_type = regularize_loss_type

        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.regularize_loss_weight = regularize_loss_weight 

        self.margin = margin

        self.tangent_enabled = self.model.name.split('_')[0] == 'smoothTC'
        self.tangent_step_weight = tangent_step_weight
        self.tangent_loss_weight = tangent_loss_weight
        self.tangent_norm_weight = tangent_norm_weight
        self.tangent_angle_weight = tangent_angle_weight
        self.tangent_k = tangent_k
        self.tangent_warmup_epochs = tangent_warmup_epochs
        self.tangent_ramp_epochs = tangent_ramp_epochs
        self.tangent_eps = tangent_eps
        self.tangent_dynamics = NSVMLP(nsv_dim=self.model.nsv_dim, **kwargs) if self.tangent_enabled else None

        self.annealing_list = model_annealing_list
        for s in self.annealing_list:
            self.__dict__[s[0]] = s[2]

        self.save_hyperparameters(ignore=['model'])
    
    
    def forward(self, x):

        if 'smooth' in self.model.name:
            output, latent, state_reconstructured, state, state_gt, latent_gt = self.model(x)
        elif 'base' in self.model.name:
            output, latent, state, latent_gt = self.model(x)
        else:
            output, latent = self.model(x)

        return output
    
    def _flow_map(self, z):
        return rk4_flow_map(self.tangent_dynamics, z, self.dt)

    def _tangent_scale(self):
        if not self.tangent_enabled:
            return 0.0
        if self.current_epoch < self.tangent_warmup_epochs:
            return 0.0
        if self.tangent_ramp_epochs <= 0:
            return 1.0
        progress = (self.current_epoch - self.tangent_warmup_epochs + 1) / float(self.tangent_ramp_epochs)
        return max(0.0, min(1.0, progress))

    def _tangent_loss(self, state, in_between_state, target_state, file_tuples):
        if not self.tangent_enabled or self.tangent_k <= 0 or state.shape[0] <= 1:
            return torch.as_tensor(0.0, device=self.device)
        step1_weight, step2_weight = 1.0, 0.5
        in_between_pred = self._flow_map(state)
        target_pred = self._flow_map(in_between_state)
        step_loss = step1_weight * self.loss_func(in_between_pred, in_between_state).sum([1]).mean() + step2_weight * self.loss_func(target_pred, target_state).sum([1]).mean()

        traj_ids = file_tuples[:, 0].to(state.device)
        neighbor_idx, valid_mask, _, kth_dist = build_cross_traj_knn_pairs(state.detach(), traj_ids, self.tangent_k)
        if not valid_mask.any():
            return self.tangent_step_weight * step_loss

        delta0 = state[neighbor_idx] - state[:, None, :]
        delta1_true = in_between_state[neighbor_idx] - in_between_state[:, None, :]
        delta2_true = target_state[neighbor_idx] - target_state[:, None, :]

        pair_weights = rbf_pair_weights(delta0, kth_dist, self.tangent_eps) * valid_mask.to(state.dtype)
        valid_rows = valid_mask.nonzero(as_tuple=False)[:, 0]

        delta0 = delta0[valid_mask]
        delta1_true = delta1_true[valid_mask]
        delta2_true = delta2_true[valid_mask]
        pair_weights = pair_weights[valid_mask]

        _, delta1_pred = transport_secants_jvp(self._flow_map, state[valid_rows], delta0)
        _, delta2_pred = transport_secants_jvp(self._flow_map, in_between_state[valid_rows], delta1_pred)

        tangent_loss = (
            step1_weight * secant_transport_loss(delta1_pred, delta1_true, pair_weights, self.tangent_eps, self.tangent_norm_weight, self.tangent_angle_weight)
            + step2_weight * secant_transport_loss(delta2_pred, delta2_true, pair_weights, self.tangent_eps, self.tangent_norm_weight, self.tangent_angle_weight)
        )

        return self.tangent_step_weight * step_loss + self.tangent_loss_weight * tangent_loss

    def calc_Losses(self, batch, is_test=False, is_val=False):

        data, output, target, file_tuples, latent_gt, latent, state = None, None, None, None, None, None, None

        if 'smooth' in self.model.name:
            data, target, in_between,  file_tuples = batch

            _, target_latent, target_state_reconstructured, target_state, target_state_gt, target_latent_gt = self.model(target)
            _, in_between_latent, in_between_state_reconstructured, in_between_state, in_between_state_gt, in_between_latent_gt = self.model(in_between)
            output, latent, state_reconstructured, state, state_gt, latent_gt = self.model(data)

            state_clone = state.clone().detach()
            state_max = torch.max(state_clone, dim=0)[0][0]
            state_min = torch.min(state_clone, dim=0)[0][0]

            #self.log('state_max{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_max, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            #self.log('state_min{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_min, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            

            if self.reconstruct_loss_type == 'high-dim-latent':
                
                reconstruct_loss = self.loss_func(latent, latent_gt).sum([1]).mean() 

            if is_val:
                self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), reconstruct_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            # smoothness loss
            smooth_loss = torch.as_tensor(0.0, device=self.device)
            if self.smooth_loss_type == 'neighbor-distance':

                data_target_dist = variable_distance(state, target_state, False)
                smooth_loss = F.relu(data_target_dist - self.margin).mean()

            if self.smooth_loss_type == 'neighbor-distance-2':

                data_target_dist = variable_distance(state, target_state, False)
                data_between_dist = variable_distance(state, in_between_state, False)

                smooth_loss = F.relu(data_target_dist - self.margin).mean() +  F.relu(data_between_dist - self.margin/2).mean()

            if self.smooth_loss_type == 'cyclic-neighbor-distance':

                data_target_dist = variable_distance(state, target_state, True)
                smooth_loss = F.relu(data_target_dist - self.margin).mean()

            if self.smooth_loss_type == 'cyclic-neighbor-distance-2':

                data_target_dist = variable_distance(state, target_state, True)
                data_between_dist = variable_distance(state, in_between_state, True)

                smooth_loss = F.relu(data_target_dist - self.margin).mean() +  F.relu(data_between_dist - self.margin/2).mean()

            if is_val:
                self.log('smth{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), smooth_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            # regularization loss
            regularize_loss = torch.as_tensor(0.0)

            if self.regularize_loss_type == 'sinkhorn':
                # collocation points in [-1, 1]^d
                v_col = 2. * torch.rand(state.shape, device=self.device) - 1.
                regularize_loss = self.regularize_loss_func(state, v_col)

            if self.regularize_loss_type == 'sinkhorn-circle':
                # collocation points in B(0, r)
                radius = 0.8 * torch.rand(state.shape[0], device=self.device)
                theta = 2 * np.pi * torch.rand(state.shape[0], device=self.device)
                v_col = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], 1)
                regularize_loss = self.regularize_loss_func(state, v_col)

            if is_val:
                self.log('reg{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), regularize_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            tangent_loss = torch.as_tensor(0.0, device=self.device)
            if self.tangent_enabled:
                tangent_loss = self._tangent_scale() * self._tangent_loss(state, in_between_state, target_state, file_tuples)

            total_loss = self.reconstruct_loss_weight * reconstruct_loss + self.beta * (self.smooth_loss_weight * smooth_loss  \
                                                    + self.regularize_loss_weight * regularize_loss) + tangent_loss

        elif 'base' in self.model.name:
            data, target, file_tuples = batch
            output, latent, state, latent_gt = self.model(data)

            state_clone = state.clone().detach()
            state_max = torch.max(state_clone, dim=0)[0][0]
            state_min = torch.min(state_clone, dim=0)[0][0]

            self.log('state_max{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_max, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('state_min{}'.format('_test' if is_test else '_val' if is_val else '_train'), state_min, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            total_loss = self.loss_func(latent, latent_gt).sum([1]).mean()
            self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        else:

            data, target, file_tuples = batch 

            output, latent = self.model(data)

            total_loss = self.loss_func(output, target).sum([1,2,3]).mean()
            self.log('rec{}_loss'.format('_test' if is_test else '_val' if is_val else '_train'), total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if is_test:
            self.save_outputs(data, output, target, file_tuples, latent_gt, latent, state)

        return total_loss

    def save_outputs(self, data, output, target, file_tuples, latent_gt, latent, state):
        pxl_loss = self.loss_func(output, target).mean()
        self.log('pxl_rec{}_loss'.format('_test'), pxl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    
        self.all_filepaths.extend(file_tuples.cpu().numpy())
        for idx in range(data.shape[0]):
            if 'save_prediction' in self.test_mode:
                comparison = torch.cat([data[idx,:, :, :128].unsqueeze(0),
                                        data[idx,:, :, 128:].unsqueeze(0),
                                        target[idx, :, :, :128].unsqueeze(0),
                                        target[idx, :, :, 128:].unsqueeze(0),
                                        output[idx, :, :, :128].unsqueeze(0),
                                        output[idx, :, :, 128:].unsqueeze(0)])
                mkpath(os.path.join(self.pred_log_dir, str(file_tuples[idx][0].item())))
                save_image(comparison.cpu(), os.path.join(self.pred_log_dir,  tuple2name(file_tuples[idx])), nrow=1)
                self.all_path_nums.add(file_tuples[idx][0].item())

            if 'base' in self.model.name or 'smooth' in self.model.name:
                latent_tmp = latent_gt[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)

                latent_reconstructed_tmp = latent[idx].view(1, -1)[0]
                latent_reconstructed_tmp = latent_reconstructed_tmp.cpu().detach().numpy()
                self.all_reconstructed_latents.append(latent_reconstructed_tmp)

                latent_latent_tmp = state[idx].view(1, -1)[0]
                latent_latent_tmp = latent_latent_tmp.cpu().detach().numpy()
                self.all_refine_latents.append(latent_latent_tmp)
            else:
                latent_tmp = latent[idx].view(1, -1)[0]
                latent_tmp = latent_tmp.cpu().detach().numpy()
                self.all_latents.append(latent_tmp)

    def training_step(self, batch, batch_idx):
        
        train_loss = self.calc_Losses(batch)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('learning rate', self.scheduler.get_lr()[0], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return train_loss

    def validation_step(self, batch, batch_idx):

        val_loss = self.calc_Losses(batch, is_val=True)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return val_loss 
    
    def test_step(self, batch, batch_idx):

        test_loss = self.calc_Losses(batch, is_test=True)

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return test_loss

    def setup(self, stage=None):

        if stage == 'test':
            # initialize lists for saving variables and latents during testing
            self.all_filepaths = []
            self.all_latents = []
            self.all_refine_latents = []
            self.all_reconstructed_latents = []
            self.all_path_nums = set()

            self.pred_log_dir = os.path.join(self.output_dir, self.model.dataset, self.pred_log_name or "predictions", self.model.name)
            self.var_log_dir = os.path.join(self.output_dir, self.model.dataset, self.var_log_name or "variables", self.model.name)
            if 'save_prediction' in self.test_mode:
                mkdir(self.pred_log_dir)
            mkdir(self.var_log_dir)
    
    def on_test_epoch_end(self) -> None:

        if 'save_prediction' in self.test_mode:
            generate_video_directory(self.pred_log_dir, self.all_path_nums, delete_after=True)

        return super().on_test_epoch_end()
    
    def configure_optimizers(self):

        params = list(self.model.parameters())
        if self.tangent_enabled:
            params += list(self.tangent_dynamics.parameters())

        ae_optimizer = torch.optim.Adam(params, lr=self.lr)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(ae_optimizer, milestones=self.lr_schedule, gamma=self.gamma)
        
        return [ae_optimizer], [self.scheduler]

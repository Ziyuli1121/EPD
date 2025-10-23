import numpy as np
import torch
from torch_utils import persistence
import torch.nn as nn
from torch.nn.functional import silu
import torch.nn.functional as F


@persistence.persistent_class
class EPD_predictor(torch.nn.Module):
    """
    Ensemble Parallel Directions
    """
    def __init__(
        self,
        num_points              = 2, # number of inter points
        dataset_name            = None,
        img_resolution          = None,
        num_steps               = None,
        sampler_tea             = None, 
        sampler_stu             = None, 
        M                       = None,
        guidance_type           = None,      
        guidance_rate           = None,
        schedule_type           = None,
        schedule_rho            = None,
        afs                     = False,
        scale_dir               = 0,
        scale_time              = 0,
        max_order               = None,
        predict_x0              = True,
        lower_order_final       = True,
        fcn                     = False,
        alpha                   = 10,
        **kwargs
    ):
        super().__init__()
        assert sampler_stu in ['epd', 'ipndm']
        assert sampler_tea in ['heun', 'dpm', 'dpmpp', 'euler', 'ipndm']
        assert scale_dir >= 0
        assert scale_time >= 0
        self.dataset_name = dataset_name
        self.img_resolution = img_resolution
        self.num_steps = num_steps
        self.sampler_stu = sampler_stu
        self.sampler_tea = sampler_tea
        self.M = M
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.scale_dir = scale_dir
        self.scale_time = scale_time
        self.max_order = max_order
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.num_points = num_points
        self.fcn = fcn
        self.alpha = alpha

        self.r_params = nn.Parameter(torch.randn(num_steps-1, num_points)) 
        self.scale_dir_params = nn.Parameter(torch.randn(num_steps-1, num_points))
        self.scale_time_params = nn.Parameter(torch.randn(num_steps-1, num_points))
        self.weight_s = nn.Parameter(torch.randn(num_steps-1, num_points))

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, batch_size, step_idx, class_labels=None,):
        weight = self.weight_s[step_idx]
        weight = weight.repeat(batch_size, 1)
        weight = self.softmax(weight)
 
        r = self.r_params[step_idx]
        r = r.repeat(batch_size, 1)
        r = self.sigmoid(r)

        params = []

        if self.scale_dir:
            scale_dir = self.scale_dir_params[step_idx]
            scale_dir = scale_dir.repeat(batch_size, 1)
            scale_dir = 2 * self.sigmoid(0.5 * scale_dir) * self.scale_dir + (1 - self.scale_dir)
            params.append(scale_dir)

        if self.scale_time:
            scale_time = self.scale_time_params[step_idx]
            scale_time = scale_time.repeat(batch_size, 1)
            scale_time = 2 * self.sigmoid(0.5 * scale_time) * self.scale_time + (1 - self.scale_time)
            params.append(scale_time)

        params.append(weight)

        return (r, *params) if params else r
    

@persistence.persistent_class
class NoiseEnsemblePredictor(torch.nn.Module):
    """Predicts noise magnitudes and aggregation weights for each timestep."""

    def __init__(
        self,
        num_points=4,
        dataset_name=None,
        img_resolution=None,
        num_steps=None,
        sampler_tea=None,
        sampler_stu=None,
        guidance_type=None,
        guidance_rate=None,
        schedule_type=None,
        schedule_rho=None,
        afs=False,
        noise_max=0.1,
        noise_damping=1.0,
        scale_dir=0,
        scale_time=0,
        max_order=None,
        predict_x0=True,
        lower_order_final=True,
        fcn=False,
        alpha=10,
        **kwargs,
    ):
        super().__init__()
        assert num_steps is not None and num_steps >= 1
        self.dataset_name = dataset_name
        self.img_resolution = img_resolution
        self.num_steps = num_steps
        self.sampler_stu = sampler_stu
        self.sampler_tea = sampler_tea
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.noise_max = noise_max
        self.noise_damping = noise_damping
        self.scale_dir = scale_dir
        self.scale_time = scale_time
        self.max_order = max_order
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.num_points = num_points
        self.fcn = fcn
        self.alpha = alpha

        self.sigma_params = nn.Parameter(torch.zeros(num_steps - 1, num_points))
        self.weight_params = nn.Parameter(torch.zeros(num_steps - 1, num_points))

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, batch_size, step_idx, class_labels=None):
        if isinstance(step_idx, torch.Tensor):
            step_idx = int(step_idx.item())

        sigma = self.sigmoid(self.sigma_params[step_idx]) * self.noise_max
        sigma = sigma.repeat(batch_size, 1)

        logits = self.weight_params[step_idx]
        logits = logits.repeat(batch_size, 1)

        if self.noise_damping > 0:
            logits = logits - self.noise_damping * (sigma ** 2)

        weights = self.softmax(logits)

        return sigma, weights

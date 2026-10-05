import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import gpytorch

from .encoder import GNN_Encoder

from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# Custom FS mol
import sys
sys.path.append("../")
from .gp_utils import ExactGPLayer
from .DNM_models2 import *
from .heads import build_head, matched_hidden

class BONDModel(nn.Module):
    def __init__(self, args):
        super(BONDModel, self).__init__()

        self.gp_kernel = "matern"
        self.emb_dim = args.emb_dim
        self.gpu_id = args.gpu_id

        self.fc_in_dim=300
        self.fc_out_dim=2048
        # The 2048-bit ECFP4 fingerprint is concatenated to the 300-d GIN embedding before the
        # head (2348 input dimensions). --use_fingerprints 0 uses the GIN embedding alone, as in
        # ADKF-IFT.
        self.use_fingerprints = getattr(args, 'use_fingerprints', 1)
        # --use_gin 0 zeroes the GIN embedding, leaving the fingerprint as the only
        # molecule-dependent input.
        self.use_gin = getattr(args, 'use_gin', 1)
        dnm_in = self.fc_in_dim + (2048 if self.use_fingerprints else 0)
        # Branches per output coordinate. With fingerprints, the DNM head contains
        # Linear(2348 -> 2048*M), about 144M parameters at M=30 (the GIN encoder has ~1.9M).
        self.dnm_M = getattr(args, 'dnm_M', 30)
        # Feature map between the encoder and the GP (see heads.py). With mlp_match_params,
        # the 'mlp' head is sized to the parameter count of the DNM head.
        self.head_type = getattr(args, 'head_type', 'dnm')
        hidden = matched_hidden(dnm_in, self.fc_out_dim, self.dnm_M) if getattr(args, 'mlp_match_params', 1) else None
        self.dnm = build_head(self.head_type, dnm_in, self.fc_out_dim,
                              M=self.dnm_M, mlp_hidden=hidden)

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.gpu_id)
        else:
            print('not load pretrained model')
        self.__create_tail_GP(kernel_type=self.gp_kernel)
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=self.device), requires_grad=True)

    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.startswith("gp_"):
            #if name.startswith("mol_encoder.gnn.gnns.4") or name.startswith("mol_encoder.gnn.batch_norms.4"):
                fe_params.append(param)
        return fe_params

    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_"):
                gp_params.append(param)
        return gp_params

    def reinit_gp_params(self, gp_input, kernel_init=None, use_lengthscale_prior=True):
        self.__create_tail_GP(kernel_type=self.gp_kernel)

        if self.gp_kernel in ['matern', 'rbf', 'RBF']:
            # 1. Current task median trick
            median_lengthscale = self.compute_median_lengthscale_init(gp_input)
            outputscale = torch.tensor(1.0, device=gp_input.device)
            if kernel_init is not None:
                if isinstance(kernel_init, dict):
                    lengthscale_init = torch.tensor(kernel_init['lengthscale'], device=gp_input.device)
                    outputscale_init = torch.tensor(kernel_init['outputscale'], device=gp_input.device)
                else:
                    lengthscale_init = torch.tensor(kernel_init[0], device=gp_input.device)
                    outputscale_init = torch.tensor(kernel_init[1], device=gp_input.device)
            else:
                lengthscale_init = median_lengthscale
                outputscale_init = outputscale
            mix_lengthscale = self.alpha * median_lengthscale + (1-self.alpha) * lengthscale_init
            mix_outputscale = self.alpha * outputscale + (1-self.alpha) * outputscale_init
            if use_lengthscale_prior:
                scale = 0.25
                loc = torch.log(mix_lengthscale).item() + scale**2
                lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )

            self.gp_model.covar_module.base_kernel.lengthscale.data.copy_(mix_lengthscale.expand_as(self.gp_model.covar_module.base_kernel.lengthscale))
            self.gp_model.covar_module.outputscale.data.copy_(mix_outputscale.expand_as(self.gp_model.covar_module.outputscale))

    def __create_tail_GP(self, kernel_type):
        dummy_train_x = torch.ones(20, self.emb_dim)
        dummy_train_y = torch.ones(20)

        ard_num_dims = None

        scale = 0.25
        loc = np.log(0.1) + scale**2 # make sure that mode=0.1
        noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior).to(self.device)
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims, use_numeric_labels=False
        ).to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model).to(self.device)

    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input) ** 2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

    def get_kernel_params(self):
        lengthscale = self.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        outputscale = self.gp_model.covar_module.outputscale.detach().cpu().numpy()
        return {
            'lengthscale': lengthscale,
            'outputscale': outputscale
        }


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        s_data,                 
        q_data,                
        task_id: str,           
        train_loss: bool,      
        kernel_init: List[float] = None,  
        s_label=None,          
        q_pred_adj: bool = False,
        predictive_val_loss: bool = False,
        is_functional_call: bool = False,
    ):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []

        fings_s = s_data.fingerprints.view(-1, 2048)

        support_features_flat, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        if not self.use_gin:
            support_features_flat = support_features_flat * 0.0
        support_labels_converted = self.__convert_bool_labels(s_label)

        support_features.append(support_features_flat)
        if self.use_fingerprints:
            support_features.append(fings_s)  # fp

        support_features = torch.cat(support_features, dim=1)  # [N, emb_dim (+2048)]
        support_features = self.dnm(support_features)

        if q_data is not None:
            fings_q = q_data.fingerprints.view(-1, 2048)
            query_features_flat, _ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            if not self.use_gin:
                query_features_flat = query_features_flat * 0.0
            query_labels_converted = self.__convert_bool_labels(q_data.y)

            query_features.append(query_features_flat)
            if self.use_fingerprints:
                query_features.append(fings_q)  # fp

            query_features = torch.cat(query_features, dim=1)  # [N, emb_dim (+2048)]
            query_features = self.dnm(query_features)

        # compute train/val loss if the model is in the training mode
        assert self.training
        assert train_loss is not None
        if train_loss: # compute train loss (on the support set)
            if is_functional_call: # return loss directly
                self.gp_model.set_train_data(inputs=support_features, targets=support_labels_converted, strict=False)
                logits = self.gp_model(support_features)
                logits = -self.mll(logits, self.gp_model.train_targets)
            else:
                self.reinit_gp_params(support_features.detach(), kernel_init, use_lengthscale_prior=True)
                self.gp_model.set_train_data(inputs=support_features.detach(), targets=support_labels_converted.detach(), strict=False)
                logits = None
        else: # compute val loss (on the query set)
            assert is_functional_call == True
            if predictive_val_loss:
                self.gp_model.eval()
                self.gp_likelihood.eval()
                with gpytorch.settings.detach_test_caches(False):
                    self.gp_model.set_train_data(inputs=support_features, targets=support_labels_converted, strict=False)
                    # return sum of the log predictive losses for all data points, which converges better than averaged loss
                    logits = -self.gp_likelihood(self.gp_model(query_features)).log_prob(query_labels_converted) #/ self.predictive_targets.shape[0]
                self.gp_model.train()
                self.gp_likelihood.train()
            else:
                # self.gp_model.set_train_data(inputs=query_features_flat, targets=query_labels_converted, strict=False)
                # logits = self.gp_model(query_features_flat)
                # logits = -self.mll(logits, self.gp_model.train_targets)
                raise NotImplementedError

        return logits


    def forward_query_loader(self, s_data, q_loader, train_loss: bool, task_id=None, s_label=None, q_pred_adj=False, predictive_val_loss: bool=False, is_functional_call: bool=False):

        # compute train/val loss if the model is in the training mode
        assert self.training == False
        support_features: List[torch.Tensor] = []
        

        fings_s = s_data.fingerprints.view(-1, 2048)

        support_features_flat, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        if not self.use_gin:
            support_features_flat = support_features_flat * 0.0
        support_labels_converted = self.__convert_bool_labels(s_label)

        support_features.append(support_features_flat)
        if self.use_fingerprints:
            support_features.append(fings_s)  # fp
        support_features = torch.cat(support_features, dim=1)
        support_features = self.dnm(support_features)

        logits_list = []
        query_labels_converted_list = []

        # do GP posterior inference if the model is in the evaluation mode
        assert train_loss is None

        self.gp_model.set_train_data(inputs=support_features, targets=support_labels_converted, strict=False)

        for q_data in q_loader:
            query_features_loader: List[torch.Tensor] = []
            q_data = q_data.to(support_features.device)
            query_labels_converted_list.append(q_data.y) # NO NEED TO CONVERT to -1, 1

            fings_q = q_data.fingerprints.view(-1, 2048)
            query_features_flat, _ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            if not self.use_gin:
                query_features_flat = query_features_flat * 0.0

            query_features_loader.append(query_features_flat)
            if self.use_fingerprints:
                query_features_loader.append(fings_q)  # fp

            query_features_predic = torch.cat(query_features_loader, dim=1)
            query_features_predic = self.dnm(query_features_predic)

            logits = self.gp_likelihood(self.gp_model(query_features_predic)).mean
            logits_list.append(logits)

        return torch.sigmoid(torch.cat(logits_list, 0)), torch.cat(query_labels_converted_list, 0)


    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0

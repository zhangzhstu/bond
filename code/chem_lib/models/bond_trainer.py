import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc, average_precision
import csv
from torch_geometric.data import DataLoader

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

from botorch.optim.fit import fit_gpytorch_scipy
from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy
from datetime import datetime
from .encoder import GNN_Encoder
# Custom FS mol
import sys
sys.path.append("../")
from .cauchy_hypergradient import cauchy_hypergradient
from ._stateless import functional_call

class BOND_Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(BOND_Meta_Trainer, self).__init__()

        self.args = args

        #self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.model = model
        self.optimizer = optim.AdamW(self.model.feature_extractor_params(), lr=args.meta_lr, weight_decay=args.weight_decay)
        #self.criterion = nn.CrossEntropyLoss().to(args.device)

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task
        self.feature_type = args.feature_type
        self.update_step = args.update_step  # the number of outer loops in an epoch during meta-training
        self.update_step_test = args.update_step_test  # the number of outer loops in an epoch during meta-testing
        #self.inner_update_step = args.inner_update_step
        now = datetime.now()
        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn + self.feature_type + now.strftime("%Y-%m-%d %H:%M:%S")
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        self.n_singular = 0   # singular-kernel failures caught in training and evaluation
        self.log_geometry = getattr(args, 'log_geometry', 0)  # --log_geometry, see _write_geometry_row
        
        self.res_logs=[]

        self.task_embedding_bank = []
        self.task_kernel_bank = []

        self.bank_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)

        model_file = args.pretrained_weight_path
        print('Bank load pretrained model from', model_file)
        self.bank_encoder.from_pretrained(model_file, args.gpu_id)
        self.task_embedding_bank = dict()
        self.task_kernel_bank = dict()

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            protocol = getattr(self.args, 'eval_protocol', 'legacy')
            s_data, q_data, q_data_adapt, sample_stats = sample_test_datasets(
                dataset, self.test_dataset, task, self.n_shot_test, self.n_query,
                self.update_step_test, protocol=protocol, return_stats=True)
            
            s_data = self.loader_to_samples(s_data)
            # q_data, fps_q = self.loader_to_samples(q_data)
            # q_data, fps_q_fit = self.loader_to_samples(q_data_adapt)

            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            # With update_step_test=0 there are no adaptation molecules and test_step skips the
            # adaptation loop, so no loader is needed; a DataLoader over an empty dataset would
            # raise in RandomSampler.
            if len(q_data_adapt) > 0:
                q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)
            else:
                q_loader_adapt = []

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'data_loader': q_loader,
                         'stats': sample_stats}

        return adapt_data, eval_data

    def compute_task2vec_embedding(self, mol_encoder, s_data, device):
        mol_encoder.eval()
        s_data = s_data.to(device)
        mol_encoder = mol_encoder.to(device)
        # forward获得特征
        feats, _ = mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)  # [N, D]
        # 用简单mean pooling/mean+std作为Task embedding
        task_embedding = torch.cat([feats.mean(0), feats.std(0)])
        return task_embedding.detach().cpu().numpy()


    def get_kernel_init_from_bank(self, support_data):
        # 计算当前任务embedding
        emb = self.compute_task2vec_embedding(self.bank_encoder, support_data, device=self.device)
        # 取所有 bank embedding 组成 (N, D) 数组
        all_embs = np.stack(list(self.task_embedding_bank.values()), axis=0)
        task_id_list = list(self.task_embedding_bank.keys())
        # 距离
        dists = np.linalg.norm(all_embs - emb, axis=1)
        best_idx = np.argmin(dists)
        best_task_id = task_id_list[best_idx]
        kernel_init = self.task_kernel_bank[best_task_id]
        return kernel_init



    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        task2id = task_id_list
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db


        for k in range(self.update_step):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            grad_accum = [0.0 for p in self.model.feature_extractor_params()]
            losses_eval = []

            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]

                if self.train_epoch > 1:
                    # kernel_init = self.get_kernel_init_from_bank(train_data['s_data'])
                    kernel_init = None
                else:
                    kernel_init = None

                # inner update
                self.model.train()
                _ = self.model(
                    s_data=train_data['s_data'], q_data=None, task_id=task_id, kernel_init=kernel_init, s_label=train_data['s_label'], train_loss=True)
                fit_gpytorch_scipy(self.model.mll)


                if task_id not in self.task_embedding_bank:
                    emb = self.compute_task2vec_embedding(self.bank_encoder, train_data['s_data'], device=self.device)
                    self.task_embedding_bank[task_id] = emb
                    kernel_params = self.model.get_kernel_params()
                    self.task_kernel_bank[task_id] = kernel_params

                # (outer) hypergrad computation
                self.model.train()
                feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if not n.startswith("gp_")]
                #feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("mol_encoder.gnn.gnns.4") or n.startswith("mol_encoder.gnn.batch_norms.4")]
                gp_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("gp_")]
                #assert False
                def f_inner(params_outer, params_inner):
                    feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                    gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                    self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                    batch_loss = functional_call(
                        self.model, self_params_dict, (train_data['s_data'], train_data['q_data']),
                        kwargs={"task_id":task_id, "s_label": train_data['s_label'], "train_loss": True, "is_functional_call": True})
                    return batch_loss

                def f_outer(params_outer, params_inner):
                    feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                    gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                    self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                    batch_loss = functional_call(
                        self.model, self_params_dict, (train_data['s_data'], train_data['q_data']),
                        kwargs={"task_id":task_id, "s_label": train_data['s_label'], "train_loss": False, "predictive_val_loss": True, "is_functional_call": True})
                    return batch_loss
                #assert False
                # The hypergradient solves a linear system in the inner-objective Hessian H, which
                # raises if H is singular (e.g. when the support kernel matrix degenerates). Count
                # the failure and skip this task's contribution instead of aborting the run.
                # --log_geometry: H is logged before the solve, see _write_geometry_row.
                hessian_callback = None
                if self.log_geometry:
                    hessian_callback = lambda H: self._write_geometry_row(k, task_id, gp_params_names, H)
                try:
                    batch_loss = cauchy_hypergradient(f_outer, f_inner, tuple(self.model.feature_extractor_params()), tuple(self.model.gp_params()), self.device, ignore_grad_correction=False, hessian_callback=hessian_callback)
                except torch._C._LinAlgError as e:
                    self.n_singular += 1
                    if self.n_singular <= 5 or self.n_singular % 100 == 0:
                        print(f'[singular] epoch {self.train_epoch} task {task_id}: {e.__class__.__name__} '
                              f'(total {self.n_singular})')
                    if self.log_geometry:
                        self._write_geometry_row(k, task_id, gp_params_names, None, event='singular_skip')
                    self.optimizer.zero_grad()
                    continue
                #assert False
                losses_eval.append(batch_loss.cpu().item()/len(train_data['q_data'].x))

                for i, param in enumerate(self.model.feature_extractor_params()):
                    if param.grad is None:
                        continue
                    grad_accum[i] += param.grad.data.clone() / len(task_id_list)

            for i, param in enumerate(self.model.feature_extractor_params()):
                param.grad = grad_accum[i]

            torch.nn.utils.clip_grad_norm_(self.model.feature_extractor_params(), 1.0)
            self.optimizer.step()

            losses_eval = np.mean(losses_eval)

            print('Train Epoch:', self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval)

        return self.model

    def test_step(self, args):
        saved_state_dict = deepcopy(self.model.state_dict())
        # reset_optimizer_after_test=2: save the optimiser state here and restore it at the end.
        saved_opt_state = (deepcopy(self.optimizer.state_dict())
                           if getattr(self.args, 'reset_optimizer_after_test', 0) == 2 else None)

        step_results={'query_preds':[], 'query_labels':[], 'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            self.model.load_state_dict(saved_state_dict)
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)

            if self.update_step_test>0 and getattr(self.args, 'eval_protocol', 'legacy') != 'support_only':
                
                for i, batch in enumerate(adapt_data['data_loader']):
                    self.optimizer.zero_grad()
                    batch = batch.to(self.device)
                    s_label_adapt = adapt_data['s_label']
                    # --scramble_adapt_labels: permute the support and query labels used for
                    # adaptation; molecules, batches, number of optimiser steps and clipping are
                    # unchanged. This separates the effect of the labels from that of the extra
                    # optimiser steps. Scoring below still uses the true labels.
                    if getattr(self.args, 'scramble_adapt_labels', 0):
                        batch = batch.clone()
                        batch.y = batch.y[torch.randperm(batch.y.shape[0], device=batch.y.device)]
                        s_label_adapt = s_label_adapt[
                            torch.randperm(s_label_adapt.shape[0], device=s_label_adapt.device)]
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': s_label_adapt,
                                        'q_data': batch, 'q_label': None}

                    # inner update
                    self.model.train()
                    _ = self.model(
                        s_data=cur_adapt_data['s_data'], q_data=None, task_id=task_id, s_label=cur_adapt_data['s_label'], train_loss=True)
                    fit_gpytorch_scipy(self.model.mll)

                    # (outer) hypergrad computation
                    self.model.train()
                    feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if not n.startswith("gp_")]
                    #feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("mol_encoder.gnn.gnns.4") or n.startswith("mol_encoder.gnn.batch_norms.4")]
                    gp_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("gp_")]
                    
                    def f_inner(params_outer, params_inner):
                        feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                        gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                        self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                        batch_loss = functional_call(
                            self.model, self_params_dict, (cur_adapt_data['s_data'], cur_adapt_data['q_data']),
                            kwargs={"task_id":task_id, "s_label": cur_adapt_data['s_label'], "train_loss": True, "is_functional_call": True})
                        return batch_loss

                    def f_outer(params_outer, params_inner):
                        feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                        gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                        self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                        batch_loss = functional_call(
                            self.model, self_params_dict, (cur_adapt_data['s_data'], cur_adapt_data['q_data']),
                            kwargs={"task_id":task_id, "s_label": cur_adapt_data['s_label'], "train_loss": False, "predictive_val_loss": True, "is_functional_call": True})
                        return batch_loss
                    #assert False
                    # Same singular-kernel guard as in train_step: skip this adaptation step
                    # instead of aborting the evaluation.
                    try:
                        batch_loss = cauchy_hypergradient(f_outer, f_inner, tuple(self.model.feature_extractor_params()), tuple(self.model.gp_params()), self.device, ignore_grad_correction=False)
                    except torch._C._LinAlgError:
                        self.n_singular += 1
                        self.optimizer.zero_grad()
                        continue
                    torch.nn.utils.clip_grad_norm_(self.model.feature_extractor_params(), 1.0)
                    self.optimizer.step()

            self.model.train()
            _ = self.model(
                s_data=eval_data['s_data'], q_data=None, task_id=task_id, s_label=eval_data['s_label'], train_loss=True)
            try:
                fit_gpytorch_scipy(self.model.mll)
            except (torch._C._LinAlgError, RuntimeError):
                self.n_singular += 1  # keep the current GP hyperparameters and score anyway

            self.model.eval()
            with torch.no_grad():
                q_preds, q_labels = self.model.forward_query_loader(eval_data['s_data'], eval_data['data_loader'], train_loss=None, task_id=task_id, s_label=eval_data['s_label'])

                # if self.args.eval_support:
                #     y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                #     y_s_true = eval_data['s_label']
                #     y_score=torch.cat([y_score, y_s_score])
                #     y_true=torch.cat([y_true, y_s_true])
                auc = auroc(q_preds, q_labels, task="binary").item()
                auprc = average_precision(q_preds, q_labels.long(), task="binary").item()

            auc_scores.append(auc)
            self._write_metrics_row(task_id, auc, auprc, eval_data.get('stats', {}))

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(q_preds.cpu().numpy())
                step_results['query_labels'].append(q_labels.cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4), ', singular_skips:', self.n_singular)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        self.model.load_state_dict(saved_state_dict)

        # The adaptation steps above use self.optimizer, so AdamW's exp_avg / exp_avg_sq buffers
        # are updated with gradients from meta-test data. Restoring the weights does not restore
        # these buffers, so in the original evaluation path they carry into later train_step calls.
        #   0 = original behaviour (optimiser state kept)
        #   1 = re-create the optimiser (also discards AdamW momentum at every evaluation)
        #   2 = restore the pre-test optimiser state (keeps the meta-training momentum)
        mode = getattr(self.args, 'reset_optimizer_after_test', 0)
        if mode == 1:
            self.optimizer = optim.AdamW(self.model.feature_extractor_params(),
                                         lr=self.args.meta_lr,
                                         weight_decay=self.args.weight_decay)
        elif mode == 2 and saved_opt_state is not None:
            self.optimizer.load_state_dict(saved_opt_state)

        return self.best_auc

    def _write_metrics_row(self, task_id, auc, auprc, stats):
        path = os.path.join(self.trial_path, 'test_metrics.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['epoch', 'task', 'protocol', 'auroc', 'auprc', 'n_scored',
                            'n_pos_scored', 'n_adapt', 'n_adapt_scored', 'n_pos_adapt_scored'])
            w.writerow([self.train_epoch, self.test_tasks[task_id],
                        getattr(self.args, 'eval_protocol', 'legacy'),
                        round(auc, 6), round(auprc, 6),
                        stats.get('n_scored', ''), stats.get('n_pos_scored', ''),
                        stats.get('n_adapt', ''), stats.get('n_adapt_scored', ''),
                        stats.get('n_pos_adapt_scored', '')])

    def _write_geometry_row(self, step, task_id, gp_names, H, event='pre_solve'):
        """Append one row to <trial_path>/geometry_log.csv (--log_geometry 1).

        H is a detached copy of the inner-objective Hessian, passed by cauchy_hypergradient
        before the solve (once per meta-training task and update step), so the row is written
        even if the solve fails. After a LinAlgError, train_step adds a row with H=None,
        event='singular_skip' and empty values.

        H is in the coordinates of the inner problem, i.e. gpytorch's raw (inverse-softplus)
        parameters, not their logs. h_ll is its (lengthscale, lengthscale) entry; h_ss, h_nn,
        h_ls, h_ln and h_sn complete the (lengthscale, outputscale, noise) block so H can be
        transformed to other coordinates.

        Values are copied to the CPU as float64 numpy, so no autograd graph is built and neither
        .grad nor the random state is touched. Errors while computing values are recorded in
        `event` and a failed write is printed; nothing is raised.
        """
        cols = ['epoch', 'step', 'task', 'lengthscale', 'outputscale', 'noise',
                'r_min', 'r_max', 'r_median', 'h_ll', 'eig_min', 'eig_max', 'sv_min', 'cond',
                'h_ss', 'h_nn', 'h_ls', 'h_ln', 'h_sn', 'event']
        vals = {'epoch': self.train_epoch, 'step': step, 'task': self.train_tasks[task_id]}
        if H is not None:
            try:
                gp = self.model.gp_model
                with torch.no_grad():
                    ls = gp.covar_module.base_kernel.lengthscale.item()
                    vals['lengthscale'] = ls
                    vals['outputscale'] = gp.covar_module.outputscale.item()
                    vals['noise'] = gp.likelihood.noise.item()
                    # train_inputs holds the support features from the forward pass that built H,
                    # so r is measured on the same points as H.
                    z = gp.train_inputs[0].detach().to('cpu', torch.float64).numpy()
                    Hn = H.to('cpu', torch.float64).numpy()
                # explicit differences; |a|^2+|b|^2-2ab loses precision for near-duplicate points
                d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
                r = d[np.triu_indices(len(z), k=1)] / ls
                vals.update(r_min=r.min(), r_max=r.max(), r_median=np.median(r))
                il, i_s, i_n = (next(i for i, n in enumerate(gp_names) if n.endswith(s))
                                for s in ('raw_lengthscale', 'raw_outputscale', 'raw_noise'))
                Hs = 0.5 * (Hn + Hn.T)
                vals.update(h_ll=Hs[il, il], h_ss=Hs[i_s, i_s], h_nn=Hs[i_n, i_n],
                            h_ls=Hs[il, i_s], h_ln=Hs[il, i_n], h_sn=Hs[i_s, i_n])
                if np.isfinite(Hn).all():
                    eig = np.linalg.eigvalsh(Hs)               # ascending
                    sv = np.linalg.svd(Hn, compute_uv=False)  # descending; H as solved
                    vals.update(eig_min=eig[0], eig_max=eig[-1], sv_min=sv[-1],
                                cond=sv[0] / sv[-1] if sv[-1] > 0 else float('inf'))
            except Exception as e:
                event = f'{event}_error:{e.__class__.__name__}'
        vals['event'] = event
        path = os.path.join(self.trial_path, 'geometry_log.csv')
        try:
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(cols)
                w.writerow([f'{v:.8g}' if isinstance(v, float) else v
                            for v in (vals.get(c, '') for c in cols)])
                fh.flush()
        except OSError as e:
            print(f'[geometry] could not write {path}: {e}')

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)

import sys
import logging

import torch
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data as data

import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

import os
import pickle

from .search import *
from .recovery import *
from .utils import *


__all__ = ["Trainer"]


class Trainer:
    def __init__(self, cfg, logging, writer):
        self.cfg = cfg

        self.num_ops      = len(PRIMITIVES)
        self.proj_dims    = cfg.proj_dims
        self.sparseness   = cfg.sparseness
        self.steps        = cfg.steps

        self.dataset      = cfg.dataset

        self.search_trainer = InnerTrainer(cfg)
        self.num_edges = self.search_trainer.model.num_edges
     
        self.save_path = cfg.save_path
        self.epochs = cfg.epochs
        
        self.logging = logging
        self.writer = writer
        self.args = cfg
        
        self.val_type = cfg.val_type
        



    def do_recovery(self, As, alpha, x_last=None, freeze_flag=None):
        xs = []
        for i in range(self.steps):
            if freeze_flag is not None and freeze_flag[i]:
                xs.append(x_last[i])
                continue
            lasso = LASSO(As[i].cpu().numpy().copy())
            b = alpha[i]
            x = lasso.solve(b)
            xs.append(x)

        return xs


    def do_search(self, A_normal, normal_biases, normal_freeze_flag,
                       A_reduce, reduce_biases, reduce_freeze_flag, epoch, all_freeze, train_loader, val_loader, test_loader, trlog):


        if not all_freeze:
            self.search_trainer.model.init_proj_mat(A_normal, A_reduce)
            self.search_trainer.model.freeze_alpha(normal_freeze_flag, reduce_freeze_flag)
            self.search_trainer.model.init_bias(normal_biases, reduce_biases)

        train_acc, train_obj = self.search_trainer.train_epoch(
             train_loader, val_loader, epoch, all_freeze, self.val_type, self.writer, self.logging)   
            
  
        self.logging.info("train_acc {:.4f}".format(train_acc))

        trlog['train_loss'].append(train_obj)
        trlog['train_acc'].append(train_acc)
        
        
        # valid
        #if (all_freeze == True and epoch >= (self.epochs - 10)) or (epoch % 50 == 0):
        valid_acc, valid_obj = self.search_trainer.validate(test_loader, all_freeze, self.val_type, epoch, trlog, self.logging)

        trlog['val_loss'].append(valid_obj)
        trlog['val_acc'].append(valid_acc)
        
        self.logging.info("valid_acc {:.4f}".format(valid_acc))
        
        self.writer.add_scalar('data/val_loss', float(valid_obj), epoch)
        self.writer.add_scalar('data/val_acc_top1', float(valid_acc), epoch)
        
        #if valid_acc > trlog['max_acc']:
        #    trlog['max_acc'] = valid_acc
        #    trlog['max_acc_epoch'] = epoch
        #    save_model('max_epoch')

        #name = 'max_acc'
        #save(self.search_trainer.model, os.path.join(self.save_path, name + '.pth'))


        if not all_freeze:
            alphas = self.search_trainer.model.arch_parameters()
            alpha_normal = torch.stack(alphas[:self.steps]).detach().cpu().numpy()
            alpha_reduce = torch.stack(alphas[self.steps:]).detach().cpu().numpy()
            return alpha_normal, alpha_reduce



    def sample_and_proj(self, base_As, xs):
        As= []
        biases = []
        for i in range(self.steps):
            A = base_As[i].numpy().copy()
            E = A.T.dot(A) - np.eye(A.shape[1])
            x = xs[i].copy()
            zero_idx = np.abs(x).argsort()[:-self.sparseness]
            x[zero_idx] = 0.
            A[:, zero_idx] = 0.
            As.append(torch.from_numpy(A).float())
            E[:, zero_idx] = 0.
            bias = E.T.dot(x).reshape(-1, self.num_ops)
            biases.append(torch.from_numpy(bias).float())

        biases = torch.cat(biases)

        return As, biases

    def show_selected(self, epoch, x_normals_last, x_reduces_last,
                                   x_normals_new, x_reduces_new, logging):
        logging.info("\n[Epoch {}]".format(epoch if epoch > 0 else 'initial'))
        # print("x_normals:\n", x_normals)
        # print("x_reduces:\n", x_reduces)
        logging.info("x_normals distance:")
        normal_freeze_flag = []
        reduce_freeze_flag = []
        for i, (x_n_b, x_n_a) in enumerate(zip(x_normals_last, x_normals_new)):
            dist = np.linalg.norm(x_n_a - x_n_b, 2)
            normal_freeze_flag.append(False if epoch == 0 else dist <= 1e-3)
            logging.info("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if normal_freeze_flag[-1] else "active"))
        logging.info("x_reduces distance:")
        for i, (x_r_b, x_r_a) in enumerate(zip(x_reduces_last, x_reduces_new)):
            dist = np.linalg.norm(x_r_a - x_r_b, 2)
            reduce_freeze_flag.append(False if epoch == 0 else dist <= 1e-3)
            logging.info("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if reduce_freeze_flag[-1] else "active"))

        logging.info("normal cell:")
        gene_normal = []
        for i, x in enumerate(x_normals_new):
            id1, id2 = np.abs(x).argsort()[-2:]
            logging.info("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_normal.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_normal.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        logging.info("reduction cell:")
        gene_reduce = []
        for i, x in enumerate(x_reduces_new):
            id1, id2 = np.abs(x).argsort()[-2:]
            logging.info("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_reduce.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_reduce.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        concat = range(2, 2 + len(x_normals_new))
        genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        logging.info(genotype)
        model_cifar = NetworkCIFAR(36, 10, 20, True, genotype)
        param_size = count_parameters_in_MB(model_cifar)
        logging.info('param size = %fMB', param_size)

        return normal_freeze_flag, reduce_freeze_flag, param_size

    def train(self, train_loader, val_loader, test_loader):
        
        #def save_model(name):
        #    torch.save(dict(params=self.model.state_dict()), osp.join(args.save_path, name + '.pth'))
        
        #trlog = {}
        #trlog['args'] = vars()
        #trlog['max_acc'] = 0.0
        #trlog['max_acc_epoch'] = 0
        
        base_A_normals = []
        base_A_reduces = []

        for i in range(self.steps):
            base_A_normals.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))
            base_A_reduces.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))

        alpha_normal = torch.stack(self.search_trainer.model.alphas_normal_).detach().cpu().numpy()
        alpha_reduce = torch.stack(self.search_trainer.model.alphas_reduce_).detach().cpu().numpy()
        x_normals_new = self.do_recovery(base_A_normals, alpha_normal)
        x_reduces_new = self.do_recovery(base_A_reduces, alpha_reduce)

        x_normals_last = x_normals_new.copy()
        x_reduces_last = x_reduces_new.copy()

        normal_freeze_flag, reduce_freeze_flag, _ = self.show_selected(
            0, x_normals_last, x_reduces_last, x_normals_new, x_reduces_new, logging)


        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['test_loss'] = []

        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['test_acc'] = []

        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        trlog['max_acc_last10'] = 0.0
        trlog['max_acc_last10_epoch'] = 0



        for i in range(self.cfg.epochs):
            A_normals, normal_biases = self.sample_and_proj(
                base_A_normals, x_normals_last)
            A_reduces, reduce_biases = self.sample_and_proj(
                base_A_reduces, x_reduces_last)
            
            self.logging.info("\nDoing Search ...")
            self.search_trainer.model.drop_path_prob = 0 #self.cfg.drop_path_prob * i / self.cfg.epochs
            alpha_normal, alpha_reduce = self.do_search(
                A_normals, normal_biases, normal_freeze_flag,
                A_reduces, reduce_biases, reduce_freeze_flag, i+1, False, train_loader, val_loader, test_loader, trlog)
            if False not in normal_freeze_flag and False not in reduce_freeze_flag:
                break
            
            self.logging.info("Doing Recovery ...")
            x_normals_new = self.do_recovery(base_A_normals, alpha_normal,
                    x_normals_last, normal_freeze_flag)
            x_reduces_new = self.do_recovery(base_A_reduces, alpha_reduce,
                    x_reduces_last, reduce_freeze_flag)
            ## update freeze flag
            normal_freeze_flag, reduce_freeze_flag, param_size = self.show_selected(
                i+1, x_normals_last, x_reduces_last, x_normals_new, x_reduces_new, logging)
            if param_size >= 3.7: # large model may cause out of memory !!!
                self.logging.info('-------------> rejected !!!')
                continue
            
            x_normals_last = x_normals_new
            x_reduces_last = x_reduces_new



        self.logging.info("\n --- Architecture Fixed, Retrain for {} Epochs --- \n".format(self.cfg.epochs))
        for i in range(self.cfg.epochs):
            self.search_trainer.model.drop_path_prob = self.cfg.drop_path_prob * i / self.cfg.epochs
            # drop path probability = 0.2
            self.do_search(
                A_normals, normal_biases, normal_freeze_flag,
                A_reduces, reduce_biases, reduce_freeze_flag, i+1, True, train_loader, val_loader, test_loader, trlog)

        self.logging.info('best epoch {}, best val acc={:.4f}, best val acc last10epoch={}, val acc last10={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], trlog['max_acc_last10_epoch'], trlog['max_acc_last10']))
            
        self.logging.info(self.save_path)








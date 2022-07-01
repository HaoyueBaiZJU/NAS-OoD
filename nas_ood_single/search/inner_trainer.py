import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .optimizer import Adam
from .models import NetWork

from .models import Generator

from .models import SinkhornDistance
from .models import domainPre
from .models import categoryPre

from ..utils import *


from torch.autograd import Variable

import numpy as np

import os

from keras.utils.np_utils import *


#from layers import SinkhornDistance



__all__ = ["InnerTrainer"]


class InnerTrainer:
    def __init__(self, cfg):
        self.auxiliary = cfg.auxiliary
        self.auxiliary_weight = cfg.auxiliary_weight

        self.grad_clip = cfg.grad_clip
        self.report_freq = cfg.report_freq
        self.save_path = cfg.save_path
        self.max_epoch = cfg.epochs
                
        self.model = NetWork(cfg.dataset, cfg.init_channels, cfg.num_classes, cfg.layers, proj_dims=cfg.proj_dims, 
                             auxiliary=cfg.auxiliary).cuda()
        self.num_concept = cfg.num_concept
        self.lambda_cycle = cfg.lambda_cycle        
        if cfg.dataset == 'finger':
            self.in_dim = 2
        else:
            self.in_dim = 3
            
        self.generator = Generator(in_dim = self.in_dim, conv_dim=64, c_dim=self.num_concept, repeat_num=3).cuda()
        self.domainPre = domainPre(cfg).cuda()
        self.categoryPre = categoryPre(cfg).cuda()
        
        self.start_epoch = cfg.start_epoch
        self.ratio = cfg.ratio

        self.lambda_lr = cfg.lambda_lr
        
        self.lambda_ot = cfg.lambda_ot
        self.lambda_ce = cfg.lambda_ce
        self.stage1_ratio = cfg.stage1_ratio        
        

        print("Param size = {}MB".format(count_parameters_in_MB(self.model)))

        weights = []
        for k, p in self.model.named_parameters():
            #if 'alpha' not in k:
            if ('alpha' not in k) and ('main' not in k) and ('input' not in k):
                weights.append(p)
        self.w_optimizer = optim.SGD(
            weights, cfg.learning_rate,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer, float(cfg.epochs), eta_min=cfg.learning_rate_min
        )
        
        self.alpha_optimizer = Adam(self.model.arch_parameters(),
            lr=cfg.arch_learning_rate, betas=(0.5, 0.999), weight_decay=cfg.arch_weight_decay)
        self.G_optimizer = Adam(self.generator.parameters(), lr=self.lambda_lr, betas=(0.5, 0.999))



    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



    def train_epoch(self, train_queue, val_queue, epoch, all_freeze, val_type, writer, logging):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss()
        criterion_cycle = nn.L1Loss().cuda()   
        sinkhorn = SinkhornDistance(eps=1, max_iter=100, reduction=None).cuda()
          

        lr = self.scheduler.get_lr()
        print('epoch: ', epoch, 'lr:', lr)

        self.model.train()
        self.generator.train()
        
        global_count = 0
        
        for batch_id, (input, target, context) in enumerate(train_queue):
        
            global_count = global_count + 1
            
            if all_freeze == False:
                N = input.size(0)

                input_train = input.cuda()
                target_train = target.cuda()
            
                input_val = input.cuda()
                target_val = target.cuda()
                context_val = context

                self.alpha_optimizer.zero_grad()
                self.w_optimizer.zero_grad()
                self.G_optimizer.zero_grad()

                domain_label = torch.tensor(to_categorical(context_val, self.num_concept + 1)).cuda()
                novel = torch.ones(N) * self.num_concept
                domain_novel = torch.tensor(to_categorical(novel, self.num_concept + 1)).cuda()

                synthetic = self.generator(input_val, domain_label, domain_novel)
                recimg = self.generator(synthetic, domain_novel, domain_label)

                feat_img = self.domainPre(input_val).cuda()
                feat_syn = self.domainPre(synthetic).cuda()
            
                loss_ot, P, C = sinkhorn(feat_syn, feat_img)
            
                logits_syn = self.categoryPre(synthetic)
                loss_ce = F.cross_entropy(logits_syn, target_val)

                loss_generator = self.lambda_cycle * criterion_cycle(recimg, input_val) - self.lambda_ot*loss_ot + 
                self.lambda_ce*loss_ce
                loss_generator.backward(retain_graph=True)
                self.G_optimizer.step()
  
                combine_img = torch.cat([input_train[int(self.stage1_ratio*N):,:,:,:], synthetic[0:int(self.stage1_ratio*N),:,:,:]], dim=0)
                target_train = torch.cat([target_train[int(self.stage1_ratio*N):], target_val[0:int(self.stage1_ratio*N)]], dim=0)
            
                scores, scores_aux = self.model(combine_img, all_freeze, grad_reverse=True)
                loss = F.cross_entropy(scores, target_train)
            
                loss.backward()
                self.alpha_optimizer.step()
                self.G_optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.w_optimizer.step()
              

            else:
                N = input.size(0)

                self.w_optimizer.zero_grad()

                input_train = input.cuda()
                target_train = target.cuda()

                domain_label = torch.tensor(to_categorical(context, self.num_concept + 1)).cuda()
                novel = torch.ones(N) * self.num_concept
                domain_novel = torch.tensor(to_categorical(novel, self.num_concept + 1)).cuda()

                if epoch >= self.start_epoch:
                    syn_train = self.generator(input_train, domain_label, domain_novel)
                    input_train = torch.cat([input_train, syn_train[0:int(self.ratio*N),:,:,:]], dim=0)
                    target_train = torch.cat([target_train, target_train[0:int(self.ratio*N)]], dim=0)                
            
                scores, scores_aux = self.model(input_train, all_freeze, grad_reverse=False)
                loss = F.cross_entropy(scores, target_train)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.w_optimizer.step()

                
            n = input_train.size(0)
            prec1, prec5 = accuracy(scores, target_train, topk=(1, 5))

            losses.update(loss.item(), n)
            
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if batch_id % self.report_freq == 0:
                logging.info("Train[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                    batch_id, losses.avg, top1.avg, top5.avg
                ))

        if all_freeze==False:
            writer.add_scalar('data/stage1_loss', float(losses.avg), epoch)
            writer.add_scalar('data/stage1_acc_top1', float(top1.avg), epoch)
            writer.add_scalar('data/stage1_acc_top5', float(top5.avg), epoch)
        else:
            writer.add_scalar('data/stage2_loss', float(losses.avg), epoch)
            writer.add_scalar('data/stage2_acc_top1', float(top1.avg), epoch)
            writer.add_scalar('data/stage2_acc_top5', float(top5.avg), epoch)

        if all_freeze:
            self.scheduler.step()    
            
        return top1.avg, losses.avg



    def validate(self, valid_queue, all_freeze, val_type, epoch, trlog, logging):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for batch_id, (input, target, context) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()

                scores, _ = self.model(input, all_freeze, grad_reverse=True)
                loss = F.cross_entropy(scores, target)

                n = input.size(0)
                prec1, prec5 = accuracy(scores, target, topk=(1, 5))
                losses.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)


                if batch_id % self.report_freq == 0:
                    logging.info(" Valid[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                        batch_id, losses.avg, top1.avg, top5.avg
                    ))

        loss = losses.avg
        acc_top1 = top1.avg
        acc_top5 = top5.avg

        if acc_top1 >= trlog['max_acc']:
            trlog['max_acc'] = acc_top1
            trlog['max_acc_epoch'] = epoch
            
        if epoch >= (self.max_epoch - 10):
            if acc_top1 >= trlog['max_acc_last10']:
                trlog['max_acc_last10'] = acc_top1
                trlog['max_acc_last10_epoch'] = epoch

                name = 'max_acc'
                torch.save(self.model.state_dict(), os.path.join(self.save_path, name + '.pth'))

                name_normal = 'max_acc_normal'
                torch.save(self.model.alphas_normal, os.path.join(self.save_path, name_normal + '.pth'))

                name_reduce = 'max_acc_reduce'
                torch.save(self.model.alphas_reduce, os.path.join(self.save_path, name_reduce + '.pth'))

                name_generator = 'max_acc_generator'
                torch.save(self.generator.state_dict(), os.path.join(self.save_path, name_generator + '.pth'))

        trlog['test_loss'].append(losses.avg)
        trlog['test_acc'].append(top1.avg)
        torch.save(trlog, os.path.join(self.save_path, 'trlog'))

        return top1.avg, losses.avg






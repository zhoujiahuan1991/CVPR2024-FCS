import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import FCSNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import os 
from scipy.spatial.distance import cdist
from torch.nn import Parameter
from torch.optim.lr_scheduler import MultiStepLR
EPSILON = 1e-8

class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

class FCS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FCSNet(args, False)
        self._protos = []
        self._covs = []
        self._radiuses = []
        init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
        self.log_dir = self.args["log_dir"]
        self.logs_name = "{}/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'],args['log_name'])
        self.logs_name = os.path.join(self.log_dir,self.logs_name)

        self.contrast_loss = SupContrastive()
        self.encoder_k = FCSNet(args, False).convnet

        self.af = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        self.save_checkpoint(os.path.join(self.logs_name,"{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"])))
    
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        task_size = self.data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size

        self._network.update_fc(self._known_classes*4,self._total_classes*4,int((task_size-1)*task_size/2))
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory(),args=self.args)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def copy_state_dict(state_dict, model, strip=None):
        tgt_state = model.state_dict()

        copied_names = set()
        for name, param in state_dict.items():

            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
            if name not in tgt_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            if param.size() != tgt_state[name].size():
                print('mismatch:', name, param.size(), tgt_state[name].size())
                continue
            tgt_state[name].copy_(param)
            copied_names.add(name)
        missing = set(tgt_state.keys()) - copied_names
        if len(missing) > 0:
            print("missing keys in state_dict:", missing)

        return model
    
    def _train(self, train_loader, test_loader):
        
        resume = False

        if self._cur_task in range(self.args["ckpt_num"]):
            p = self.args["ckpt_path"]
            detail = p.split('/')
            l = "{}_{}_{}_{}.pkl".format('fcs',detail[-3],detail[-2],self._cur_task)

            l = os.path.join(p,l)
            print('load from {}'.format(l))
            self._network.load_state_dict(torch.load(l)["model_state_dict"],strict=False)
            resume = True

        self._network.to(self._device)

        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if not resume:

            
            if self._cur_task == 0 and self.args["dataset"]=="imagenetsubset":
                self._epoch_num = self.args["epochs_init"]
                print('use {} optimizer'.format(self._cur_task))
                base_lr = 0.1  # Initial learning rate
                lr_strat = [80, 120, 150] # Epochs where learning rate gets decreased
                lr_factor = 0.1  # Learning rate decrease factor
                custom_weight_decay = 5e-4  # Weight Decay
                custom_momentum = 0.9  # Momentum
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), lr=base_lr, momentum=custom_momentum,
                                    weight_decay=custom_weight_decay)
                scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)
            else :
                self._epoch_num = self.args["epochs"]
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
    
        self._build_protos()
    
    def _build_protos(self):

        if self._cur_task != 0 :
            proto = torch.tensor(self._protos).float().cuda()
            self._network.transfer.eval()
            with torch.no_grad():
                proto_transfer = self._network.transfer(proto)["logits"].cpu().tolist()
            self._network.transfer.train()
            for i in range(len(self._protos)):
                self._protos[i]=np.array(proto_transfer[i])
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)

                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)
                
                cov = np.cov(vectors.T)
                self._covs.append(cov)
                self._radiuses.append(np.trace(cov)/vectors.shape[1])
            self._radius = np.sqrt(np.mean(self._radiuses))
    
  
    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_clf, losses_fkd, losses_proto, losses_transfer,losses_contrast\
                = 0., 0., 0. ,0., 0. 
            correct, total = 0, 0

            for i,instance  in enumerate(train_loader):

                (_, inputs, targets,inputs_aug) =instance 
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs_aug = inputs_aug.to(self._device, non_blocking=True)
                #image_q, image_k = image_q.to(
                #    self._device, non_blocking=True), image_k.to(self._device, non_blocking=True)
                    
                inputs,targets,inputs_aug = self._class_aug(inputs,targets,inputs_aug=inputs_aug)

                logits, losses_all  = self._compute_il2a_loss(inputs,targets,image_k=inputs_aug)
                loss_clf= losses_all["loss_clf"]
                loss_fkd= losses_all["loss_fkd"]
                loss_proto= losses_all["loss_proto"]
                loss_transfer= losses_all["loss_transfer"]
                loss_contrast= losses_all["loss_contrast"]
                loss = loss_clf + loss_fkd + loss_proto  + loss_transfer +loss_contrast
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                losses_transfer += loss_transfer.item()
                losses_contrast += loss_contrast.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                #break
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_transfer {:.3f}, Loss_contrast {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_transfer/len(train_loader), losses_contrast/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_transfer {:.3f}, Loss_contrast {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_transfer/len(train_loader), losses_contrast/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
  
    def l2loss(self,inputs,targets,mean=True):

        if not mean :
            delta = torch.sqrt(torch.sum( torch.pow(inputs-targets,2) ))
            return delta 
        else :
            delta = torch.sqrt(torch.sum( torch.pow(inputs-targets,2),dim=-1 ))
            
            return torch.mean(delta)
    
    @torch.no_grad()
    def _copy_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        self.encoder_k.to(self._device)
        for param_q, param_k in zip(
            self._network.convnet.parameters(),self.encoder_k.parameters() 
        ):
            param_k.data =  param_q.data
    
    def _compute_il2a_loss(self,inputs, targets,image_k=None):
        loss_clf,loss_fkd,loss_proto,loss_transfer,loss_contrast= \
            torch.tensor(0.) , torch.tensor(0.) , torch.tensor(0.) , torch.tensor(0.) , torch.tensor(0.) 
        
        network_output = self._network(inputs)
        
        features = network_output["features"]
        
        if image_k!=None and (self._cur_task==0):
            b = image_k.shape[0]
            targets_part = targets[:b].clone()
            
            with torch.no_grad():
                #a_ = self._network.convnet.layer4[1].bn1.running_mean

                self._copy_key_encoder()
                #self.encoder_k.to(self._device)
                #features_q_ = self._network(image_k)["features"]
                features_k = self.encoder_k(image_k)["features"]
                features_k = nn.functional.normalize(features_k, dim=-1)

            features_q = nn.functional.normalize(features[:b], dim=-1)

            l_pos_global = (features_q * features_k).sum(-1).view(-1, 1)

            l_neg_global = torch.einsum('nc,ck->nk', [features_q, features_k.T])
  
            # logits: Nx(1+K)
            logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)

            # apply temperature
            logits_global /= self.args["contrast_T"]

            # one-hot target from augmented image
            positive_target = torch.ones((b, 1)).cuda()
            # find same label images from label queue
            # for the query with -1, all 
            negative_targets = ((targets_part[:, None] == targets_part[None, :]) & (targets_part[:, None] != -1)).float().cuda()
            targets_global = torch.cat([positive_target, negative_targets], dim=1)

            loss_contrast = self.contrast_loss(logits_global, targets_global)*self.args["lambda_contrast"]
            
        #print(network_output.keys())
        logits = network_output["logits"]
        loss_clf = F.cross_entropy(logits/self.args["temp"], targets)

        if self._cur_task != 0:
            features_old = self.old_network_module_ptr.extract_vector(inputs)



        if self._cur_task == 0:
            losses_all = { 
            "loss_clf": loss_clf,
            "loss_fkd": loss_fkd,
            "loss_proto": loss_proto,
            "loss_transfer": loss_transfer,
            "loss_contrast": loss_contrast,
        }
            return logits, losses_all
        
        

        feature_transfer = self._network.transfer(features_old)["logits"]
        loss_transfer = self.args["lambda_transfer"] * self.l2loss(features,feature_transfer) 

      
        loss_fkd = self.args["lambda_fkd"] * self.l2loss(features,features_old,mean=False) 
        
        index = np.random.choice(range(self._known_classes),size=self.args["batch_size"],replace=True)
    
        proto_features_raw = np.array(self._protos)[index]
        proto_targets = index*4

        proto_features = proto_features_raw + np.random.normal(0,1,proto_features_raw.shape)*self._radius
        proto_features = torch.from_numpy(proto_features).float().to(self._device,non_blocking=True)
        proto_targets = torch.from_numpy(proto_targets).to(self._device,non_blocking=True)

        proto_features_transfer = self._network.transfer(proto_features)["logits"].detach().clone()
        proto_logits = self._network_module_ptr.fc(proto_features_transfer)["logits"][:,:self._total_classes*4]
    
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets)

        if image_k!=None and (self._cur_task>0 ):
            b = image_k.shape[0]
            targets_part = targets[:b].clone()
            targets_part_neg = targets[:b].clone()
            with torch.no_grad():

                self._copy_key_encoder()
                features_k = self.encoder_k(image_k)["features"]
                features_k = torch.cat((features_k,proto_features),dim=0)
                features_k = nn.functional.normalize(features_k, dim=-1)
                targets_part_neg = torch.cat((targets_part_neg,proto_targets),dim=0)
            #print(features_k.shape,targets_part_neg.shape,b,proto_features.shape)
            features_q = nn.functional.normalize(features[:b], dim=-1)
            
            l_pos_global = (features_q * features_k[:b]).sum(-1).view(-1, 1)
            l_neg_global = torch.einsum('nc,ck->nk', [features_q, features_k.T])

            # logits: Nx(1+K)
            logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)
            # apply temperature
            logits_global /= self.args["contrast_T"]

            # one-hot target from augmented image
            positive_target = torch.ones((b, 1)).cuda()
            # find same label images from label queue
            # for the query with -1, all 
            negative_targets = ((targets_part[:, None] == targets_part_neg[None, :]) & (targets_part[:, None] != -1)).float().cuda()
            targets_global = torch.cat([positive_target, negative_targets], dim=1)
            loss_contrast = self.contrast_loss(logits_global, targets_global)*self.args["lambda_contrast"]

        losses_all = { 
            "loss_clf": loss_clf,
            "loss_fkd": loss_fkd,
            "loss_proto": loss_proto,
            "loss_transfer": loss_transfer,
            "loss_contrast":loss_contrast,
        }
        
        return logits, losses_all 



    




    def _class_aug(self,inputs,targets,alpha=20., mix_time=4,inputs_aug=None):
        
        inputs2 = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
        inputs2 = inputs2.view(-1, 3, inputs2.shape[-2], inputs2.shape[-1])
        targets2 = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

        inputs_aug2 = torch.stack([torch.rot90(inputs_aug, k, (2, 3)) for k in range(4)], 1)
        inputs_aug2 = inputs_aug2.view(-1, 3, inputs_aug2.shape[-2], inputs_aug2.shape[-1])
            
                
        
        mixup_inputs = []
        mixup_targets = []

        for _ in range(mix_time):
            index = torch.randperm(inputs.shape[0])
            perm_inputs = inputs[index]
            perm_targets = targets[index]
            mask = perm_targets!= targets

            select_inputs = inputs[mask]
            select_targets = targets[mask]
            perm_inputs = perm_inputs[mask]
            perm_targets = perm_targets[mask]

            lams = np.random.beta(alpha,alpha,sum(mask))
            lams = np.where((lams<0.4)|(lams>0.6),0.5,lams)
            lams = torch.from_numpy(lams).to(self._device)[:,None,None,None].float()


            mixup_inputs.append(lams*select_inputs+(1-lams)*perm_inputs)
            mixup_targets.append(self._map_targets(select_targets,perm_targets))
            

                
        mixup_inputs = torch.cat(mixup_inputs,dim=0)
        
        mixup_targets = torch.cat(mixup_targets,dim=0)

        inputs = torch.cat([inputs2,mixup_inputs],dim=0)
        targets = torch.cat([targets2,mixup_targets],dim=0)

        return inputs,targets,inputs_aug2
    
    def _map_targets(self,select_targets,perm_targets):
        assert (select_targets != perm_targets).all()
        large_targets = torch.max(select_targets,perm_targets)-self._known_classes
        small_targets = torch.min(select_targets,perm_targets)-self._known_classes

        mixup_targets = (large_targets*(large_targets-1)/2  + small_targets + self._total_classes*4).long()
        return mixup_targets
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:,:self._total_classes*4][:,::4]
            predicts = torch.max(outputs, dim=1)[1]
                
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader,only_new,only_old):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:,:self._total_classes*4][:,::4]
                if only_new:
                    outputs[:,:self._known_classes] = -100
                if only_old:
                    outputs[:,self._known_classes:] = -100
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            
            

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def eval_task(self,only_new=False,only_old = False):
        y_pred, y_true = self._eval_cnn(self.test_loader,only_new=only_new,only_old=only_old)

        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            print(len(self._protos))
            y_pred, y_true = self._eval_nme(self.test_loader, self._protos/np.linalg.norm(self._protos,axis=1)[:,None])
            nme_accy = self._evaluate(y_pred, y_true)            
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    
    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]


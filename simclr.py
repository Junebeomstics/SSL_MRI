import logging
import os
import time
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, AverageMeter
from losses import info_nce_loss
from torch.nn import CrossEntropyLoss
from utils import LinearClassifier
import numpy as np
torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, model, loss, config, loader_train, loader_val, optimizer, scheduler, batch_size = 4, n_views = 2, temperature = 0.07):
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter()
        self.config = config
        self.loader = loader_train
        self.loader_val = loader_val
        self.temperature = temperature
        self.n_views = n_views
        self.batch_size = batch_size
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.metrics = {}
        
    def pretraining(self, train_loader):

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.config.nb_epochs} epochs.")

        for epoch_counter in range(self.config.nb_epochs):
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            
            for images, _ in tqdm(train_loader):
                pbar.update()
                images = images.to(self.device)
                self.optimizer.zero_grad()
                images = images[:,0,:]
                images = images[:, None,:,:]
                images = self.model(images)
                
                logits, labels = info_nce_loss(images, self.batch_size, self.n_views, self.temperature)

                loss = self.criterion(logits, labels)
                loss.backward()
                training_loss += float(loss) / nb_batch
                self.optimizer.zero_grad()


                if n_iter % 100 == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
            pbar.close()
            
            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_values = {}
            with torch.no_grad():
                self.model.eval()
                for (inputs, _) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    inputs = inputs[:,0, :]
                    inputs = inputs[:, None,:,:]
                    images = self.model(inputs)
                    logits, labels = info_nce_loss(images, self.batch_size, self.n_views, self.temperature)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    val_loss += float(loss) / nb_batch
                    for name, metric in self.metrics.items():
                        if name not in val_values:
                            val_values[name] = 0
                        val_values[name] += metric(logits, labels) / nb_batch
            pbar.close()

            metrics = "\t".join(["Validation {}: {:.4f}".format(m, v) for (m, v) in val_values.items()])
            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch_counter+1, self.config.nb_epochs, training_loss, val_loss)+metrics, flush=True)

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch_counter % self.config.nb_epochs_per_saving == 0 or epoch_counter == self.config.nb_epochs - 1) and epoch_counter > 0:
                pass
            if epoch_counter==99:
                torch.save({
                    "epoch": epoch_counter,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()},
                    os.path.join(self.config.checkpoint_dir, "{name}_epoch_{epoch}.pth".
                                 format(name="y-Aware_Contrastive_MRI", epoch=epoch_counter)))

        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.config.nb_epochs)
        save_checkpoint({
            'epoch': self.config.nb_epochs,
            'arch': 'resnet18',
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


    def fine_tuning(self):

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0 ### ADNI [] replaced to 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, _) in self.loader:
                pbar.update()
                #labels = torch.flatten(labels).type(torch.LongTensor) # ADNI
                
                inputs = inputs[:,0, :]
                inputs = inputs[:, None,:,:]
                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                images = self.model(inputs)
                logits, labels = info_nce_loss(images, self.batch_size, self.n_views, self.temperature)
                loss = self.criterion(logits, labels)
                
                self.optimizer.step()
                loss.backward()
                training_loss += float(loss) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            epoch_outputs = [] # ADNI
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    labels = torch.flatten(labels).type(torch.LongTensor) # ADNI
                    inputs = inputs.to(self.device)
                    inputs = inputs[:,0, :]
                    inputs = inputs[:, None,:,:]
                    labels = labels.to(self.device)
                    y = self.model(inputs)
                    m = torch.nn.Softmax(dim=1) # ADNI
                    output = m(y) # ADNI
                    epoch_outputs.append(output)
                    batch_loss = self.loss(y, labels)
                    val_loss += float(batch_loss) / nb_batch
            pbar.close()

            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss), flush=True)

            if self.scheduler is not None:
                self.scheduler.step()

        ## Test step
        nb_batch = len(self.loader_val)
        pbar = tqdm(total=nb_batch, desc="Test")
        val_loss = 0
        outGT = torch.FloatTensor().cuda() # ADNI
        outPRED = torch.FloatTensor().cuda() # ADNI
        with torch.no_grad():
            self.model.eval()
            for (inputs, labels) in self.loader_val:
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor) # ADNI
                inputs = inputs.to(self.device)
                inputs = inputs[:,0, :]
                inputs = inputs[:, None,:,:]
                labels = labels.to(self.device)
                y = self.model(inputs)
                ### ADNI
                m = torch.nn.Softmax(dim=1)
                output = m(y)
                if int(labels) == 0:
                    onehot = torch.LongTensor([[1, 0, 0]])
                elif int(labels) == 1:
                    onehot = torch.LongTensor([[0, 1, 0]])
                else:
                    onehot = torch.LongTensor([[0, 0, 1]])
                onehot = onehot.cuda()
                outGT = torch.cat((outGT, onehot), 0)
                outPRED = torch.cat((outPRED, output), 0)
                ###
                batch_loss = self.loss(y, labels)
                val_loss += float(batch_loss) / nb_batch
        pbar.close()

        return outGT, outPRED # ADNI

    def validate(self, val_loader):
        """validation"""
        classifier = LinearClassifier(name=self.config.model, num_classes= self.config.num_classes)

        self.model.eval()
        classifier.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            end = time.time()
            for idx, (images, labels) in enumerate(val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # forward
                output = classifier(model.encoder(images))
                loss = CrossEntropyLoss()

                # update metric
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % 20 == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        return losses.avg, top1.avg
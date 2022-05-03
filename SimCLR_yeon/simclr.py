import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, loss, model, config, loader_train, loader_val, optimizer, scheduler, batch_size = 4, n_views = 2, temperature = 0.07):
        self.device = torch.device("cuda")
        self.loss = loss
        self.model = model.to(self.device)
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

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        if similarity_matrix.shape == labels.shape:

            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
            # assert similarity_matrix.shape == labels.shape
    
            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
    
            logits = logits / self.temperature
        else:
            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
            # assert similarity_matrix.shape == labels.shape
    
            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
    
            logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):
        print(self.loss)
        print(self.optimizer)

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
                
                logits, labels = self.info_nce_loss(images)
                loss = self.criterion(logits, labels)
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
                    logits, labels = self.info_nce_loss(images)
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

import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging
from Earlystopping import EarlyStopping # ADNI



class yAwareCLModel:

    def __init__(self, net, loss, loader_train, loader_val, loader_test, config, task_name, train_num, stratify, scheduler=None): # ADNI
        """

        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("yAwareCL")
        self.loss = loss
        self.model = net
        ### ADNI
        print('config.mode:', config.mode)
        if config.mode == 0: # PRETRAINING
            self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        else: # config.mode == 1: # FINE_TUNING
            if config.layer_control == 'tune_all':
                self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            elif config.layer_control == 'freeze':
                self.optimizer = torch.optim.Adam(net.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            else: # config.layer_control == 'tune_diff':
                if config.model == 'DenseNet':
                    self.optimizer = torch.optim.Adam([
                        {"params": net.features.conv0.parameters(), "lr": config.lr*1e-3},
                        {"params": net.features.denseblock1.parameters(), "lr": config.lr*1e-3},
                        {"params": net.features.transition1.parameters(), "lr": config.lr*1e-3},

                        {"params": net.features.denseblock2.parameters(), "lr": config.lr*1e-2},
                        {"params": net.features.transition2.parameters(), "lr": config.lr*1e-2},

                        {"params": net.features.denseblock3.parameters(), "lr": config.lr*1e-1},
                        {"params": net.features.transition3.parameters(), "lr": config.lr*1e-1},

                        {"params": net.features.denseblock4.parameters(), "lr": config.lr},
                        {"params": net.classifier.parameters(), "lr": config.lr},
                        ], lr=config.lr, weight_decay=config.weight_decay)
                else: # config.model == 'UNet':
                    self.optimizer = torch.optim.Adam([
                        {"params": net.up.parameters(), "lr": config.lr*1e-2},
                        {"params": net.down.parameters(), "lr": config.lr*1e-1},
                        {"params": net.classifier.parameters(), "lr": config.lr},
                        ], lr=config.lr, weight_decay=config.weight_decay)
        ###

        self.scheduler = scheduler
        self.loader = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test # ADNI
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.metrics = {}
        ### ADNI
        if train_num != 0:
            self.task_name = task_name
            self.train_num = train_num
            self.stratify = stratify
        ###
        
        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            self.load_model(config.pretrained_path)

        self.model = DataParallel(self.model).to(self.device)

    def pretraining(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):

            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                z_i = self.model(inputs[:, 0, :])
                z_j = self.model(inputs[:, 1, :])
                batch_loss, logits, target = self.loss(z_i, z_j, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += batch_loss.item()*inputs.size(0) # ADNI
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_values = {}
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    z_i = self.model(inputs[:, 0, :])
                    z_j = self.model(inputs[:, 1, :])
                    batch_loss, logits, target = self.loss(z_i, z_j, labels)
                    val_loss += batch_loss.item()*inputs.size(0) # ADNI
                    for name, metric in self.metrics.items():
                        if name not in val_values:
                            val_values[name] = 0
                        val_values[name] += metric(logits, target)*inputs.size(0) # ADNI
            pbar.close()

            ### ADNI
            metrics = "\t".join(["Validation {}: {:.4f}".format(m, v) for (m, v) in val_values.items()])
            print("\nEpoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss / len(self.loader.dataset), val_loss / len(self.loader_val.dataset))+metrics, flush=True)
            ###

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1) and epoch > 0:
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()},
                    os.path.join(self.config.checkpoint_dir, "{name}_epoch_{epoch}.pth".
                                 format(name="ADNI_y-Aware_Contrastive_MRI", epoch=epoch)))

    def fine_tuning(self):
        print(self.loss)
        print(self.optimizer)
        early_stopping = EarlyStopping(patience = self.config.patience, path = './ckpts/ADNI_{0}_{2}_{1}.pt'.format(self.task_name.replace('/', ''), self.train_num, self.stratify)) # ADNI
        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0 ### ADNI [] replaced to 0 (original code had error)
            training_acc = 0 # ADNI
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor) # ADNI
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                if self.config.task_type == 'reg': # ADNI
                    labels = labels.to(torch.float32) # ADNI
                batch_loss = self.loss(y, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += batch_loss.item()*inputs.size(0) # ADNI
                if self.config.task_type == 'cls': # ADNI
                    _, predicted = torch.max(y, 1) # ADNI
                    training_acc += (predicted == labels).sum().item() # ADNI
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            val_acc = 0 # ADNI
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    labels = torch.flatten(labels).type(torch.LongTensor) # ADNI
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)
                    if self.config.task_type == 'reg': # ADNI
                        labels = labels.to(torch.float32) # ADNI
                    batch_loss = self.loss(y, labels)
                    val_loss += batch_loss.item()*inputs.size(0) # ADNI
                    if self.config.task_type == 'cls': # ADNI
                        _, predicted = torch.max(y, 1) # ADNI
                        val_acc += (predicted == labels).sum().item() # ADNI
            pbar.close()
            ### ADNI
            print("\nEpoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                  epoch+1, self.config.nb_epochs, training_loss / len(self.loader.dataset), val_loss / len(self.loader_val.dataset)))
            if self.config.task_type == 'cls': # ADNI
                print("Training accuracy: {:.2f}%\t Validation accuracy: {:.2f}%\t".format(
                      100 * training_acc / len(self.loader.dataset), 100 * val_acc / len(self.loader_val.dataset), flush=True))
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("[ Early stopped ]")
                break
            ###
            if self.scheduler is not None:
                self.scheduler.step()

        ### ADNI
        self.model.load_state_dict(torch.load('./ckpts/ADNI_{0}_{2}_{1}.pt'.format(self.task_name.replace('/', ''), self.train_num, self.stratify))) # ADNI

        ## Test step
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test")
        test_loss = 0
        test_acc = 0
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        with torch.no_grad():
            self.model.eval()
            for (inputs, labels) in self.loader_test:
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y = self.model(inputs)
                if self.config.task_type == 'reg': # ADNI
                    labels = labels.to(torch.float32) # ADNI
                    outGT = torch.cat((outGT, labels), 0) # ADNI
                    outPRED = torch.cat((outPRED, y), 0) # ADNI

                if self.config.task_type == 'cls': # ADNI
                    m = torch.nn.Softmax(dim=1)
                    output = m(y)
                    if int(labels) == 0:
                        onehot = torch.LongTensor([[1, 0]])
                    elif int(labels) == 1:
                        onehot = torch.LongTensor([[0, 1]])
                    onehot = onehot.cuda()
                    outGT = torch.cat((outGT, onehot), 0)
                    outPRED = torch.cat((outPRED, output), 0)
                    _, predicted = torch.max(y, 1)
                    test_acc += (predicted == labels).sum().item()

                batch_loss = self.loss(y, labels)
                test_loss += batch_loss.item()*inputs.size(0)
                    
        pbar.close()
        if self.config.task_type == 'cls':
            print("\n\nTest loss: {:.4f}\t Test accuracy: {:.2f}%\t".format(
                  test_loss / len(self.loader_test.dataset), 100 * test_acc / len(self.loader_test.dataset)), flush=True)
        else:
            print("\n\nTest loss: {:.4f}".format(test_loss / len(self.loader_test.dataset), flush=True))
        ###
        return outGT, outPRED # ADNI


    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        self.logger.info('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))

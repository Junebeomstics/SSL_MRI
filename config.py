
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode, framework):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode
        self.framework = framework

        if self.mode == PRETRAINING:
            self.batch_size = 64
            self.nb_epochs_per_saving = 1 # 몇 에포크에 한 번씩 저장할 것인가
            self.pin_mem = True
            self.num_cpu_workers = 8
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 5 # depends on the meta-data at hand
            self.temperature = 0.1
            self.tf = "all_tf"
            self.model = "DenseNet"
            
            self.data = '/global/cfs/cdirs/m3898/2.UKB/1.sMRI_fs_cropped' 
            self.label = '/global/cfs/cdirs/m3898/2.UKB/2.demo_qc/UKB_phenotype.csv' 
            
            self.val_size = 0.1
            # Paths to the data
#             self.data_train = "/path/to/your/training/data.npy"
#             self.label_train = "/path/to/your/training/metadata.csv"

#             self.data_val = "/path/to/your/validation/data.npy"
#             self.label_val = "/path/to/your/validation/metadata.csv"

            self.input_size = (1, 121, 145, 121)
            self.label_name = "age"

            self.checkpoint_dir = "./checkpoint_simclr"
            
            self.train_continue = "on"

        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 8
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            self.pretrained_path = "/path/to/model.pth"
            self.num_classes = 2
            self.model = "DenseNet"

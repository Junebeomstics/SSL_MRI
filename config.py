
PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode

        if self.mode == PRETRAINING:
            self.batch_size = 64
            self.nb_epochs_per_saving = 1
            self.pin_mem = True
            self.num_cpu_workers = 8
            self.nb_epochs = 100 # ADNI
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.sigma = 5 # depends on the meta-data at hand
            self.temperature = 0.1
            self.tf = "all_tf"
            self.model = "DenseNet"


            # Paths to the data
            ### ADNI
            #self.data_train = "/path/to/your/training/data.npy"
            #self.label_train = "/path/to/your/training/metadata.csv"

            #self.data_val = "/path/to/your/validation/data.npy"
            #self.label_val = "/path/to/your/validation/metadata.csv"
            ###

            self.input_size = (1, 121, 145, 121)
            self.label_name = "PTAGE" # ADNI
            self.freeze = False # ADNI
            self.checkpoint_dir = "./ckpts" # ADNI

        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 8
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100 # ADNI
            self.cuda = True
            self.tf = "cutout" # ADNI
            self.input_size = (1, 121, 145, 121) # ADNI
            self.patience = 20 # ADNI
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5

            self.pretrained_path = "./DenseNet121_BHB-10K_yAwareContrastive.pth" # ADNI
            self.freeze = True # ADNI (whether to freeze pretrained layers or not)
            self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN
            self.model = "DenseNet"

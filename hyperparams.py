from collections import namedtuple

HParams = namedtuple(
    "HParams",
    [
        #training params
    	"batch_size",
        "augment_flag",
        "num_epochs",

        #Directories/Storage
        "input_dir",
        "weights_dir",
        "weights_for_testing",
        "results_dir",

        #I/O-data: parameters
        "volume_width",
        "volume_height",
        "volume_depth",
        "num_tissues" #"Number of different label classes in the ground truth
    ])

def create_hparams():

  return HParams(
      volume_width  = 160,
      volume_height = 160,
      volume_depth  = 2, # 1 channel=LGE; 2 channels= Cine-LGE Fusion;
      num_tissues   = 4,
      # training
      batch_size     = 4, # num of patients
      num_epochs     = 250,
      augment_flag   = True,

      # Directories/Storage
      input_dir='./data/',
      results_dir='./results/',
      weights_dir='./models/',
      weights_for_testing = '/model_lge_weights-Fusion.hdf5'
  )

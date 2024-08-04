import argparse
from config_mf import DATASET_DIR, MAX_EPOCHS, BATCH_SIZE, LR, NET


BASE_DIR = "/root/work/src/carbon/"

class BaseParams(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParams, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--net",
            type=str,
            default=NET,
            help="network name"
        )

        self.add_argument(
            "--dataset",
            type=str,
            default='carbon',
            help="dataset name"
        )

        self.add_argument(
            "--dataset_base_dir",
            type=str,
            # default=BASE_DIR+'NIA_arg/',
            default=DATASET_DIR,
            help="dataset name"
        )

        self.add_argument(
            "--image_size",
            type=int,
            default=512,
            help="image size default = 512"
        )


        self.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Fixed random seed"
        )
        self.add_argument(
            "--local_rank",
            type=int,            
            # required=True,
            help='local rank for DistributedDataParallel')
        self.add_argument("--tensorboard", default=True, action="store_true")

        self.add_argument(
            "--num_workers",
            type=int,
            default=2,
            # default=0,
            help="num_workers"
        )


class TrainParser(BaseParams):
    def __init__(self):
        super(TrainParser, self).__init__()
        self.add_argument("--output-dir", default="output", type=str, help="output directory")
        # self.add_argument("--output-dir", default="outputs/outputs_carbon_220902_CRBN_QNTT_new_CabonClip2000/", type=str, help="output directory")
        self.add_argument("--train_batch_size", default=BATCH_SIZE, type=int, help="train batch size")
        self.add_argument("--val_batch_size", default=BATCH_SIZE, type=int, help="validataion batch size")
        self.add_argument("--total-epoch", default=MAX_EPOCHS, type=int, help="total num epoch")
        self.add_argument("--eval-freq", default=5, type=int, help="total num epoch")
        self.add_argument("--save-freq", default=1000, type=int, help="total num epoch")
        self.add_argument("--learning-rate", default=LR, type=float, help="learning late") #0.001 1e-5
        self.add_argument('--resume', action="store_true", help='resume from checkpoint')
        self.add_argument('--opt', default="adam", type=str, help="nadam, adam")
        self.add_argument('--lrs', default="cosinealr", type=str, help="cosinealr, steplr")
        self.add_argument('--enc_dropout', action="store_true", help='dropout for encoder')
        
class InferenceParser(BaseParams):
    def __init__(self):
        super(InferenceParser, self).__init__()
        self.add_argument("--output-dir", default="/root/work/src/carbon/outputs_carbon_220822_2_LossWeight_50_005_nan/results_best_checkpoints_loss/", type=str, help="output directory")
        self.add_argument("--val_batch_size", default=1, type=int, help="validataion batch size")
        self.add_argument("--model_path", default="/outputs_carbon_220822_2_LossWeight_50_005_nan/pths/best_checkpoints_loss.pth", type=str, help="pretrained model path")
        # self.add_argument("--image_dir", default="/home/mind2/work/project/dataset/NIA_arg/Validation", type=str, help="image directory")
        self.add_argument("--image_csv", default="/root/work/dataset/test.csv", type=str, help="test image csv")
        self.add_argument("--image_type", default="forest_AP_10_25", type=str, help="test image type")
        
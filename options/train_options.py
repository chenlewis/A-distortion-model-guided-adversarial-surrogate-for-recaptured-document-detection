from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--norm_flag', required=False,
                            default=True)
        parser.add_argument('--max_iter', required=False, type=int,
                            default=20480)
        parser.add_argument('--iter_per_epoch', required=False, type=int,
                            default=1024)
        parser.add_argument('--epoch_count', type=int, 
                            default=1)
        parser.add_argument('--batch_size', type=int, 
                            default=4)
        parser.add_argument('--cnn_model', type=str, 
                            default='ResNet50')
        parser.add_argument('--lambda_triplet', required=False, type=float,
                            default=2)
        parser.add_argument('--lambda_adgenuine', required=False, type=float,
                            default=0.1)
        parser.add_argument('--cnn_init_lr', required=False, type=float,
                            default=0.0001)      
        parser.add_argument('--surrogate_init_lr', type=float, 
                            default=0.0001)
        parser.add_argument('--src1_genuine_csv', required=True,
                            default="")  
        parser.add_argument('--src1_recaptured_csv', required=True,
                            default="")
        parser.add_argument('--src2_genuine_csv', required=True,
                            default="")
        parser.add_argument('--src2_recaptured_csv', required=True,
                            default="")
        parser.add_argument('--test_csv', required=True,
                            default="")
        parser.add_argument('--load_cnn_path', required=False,
                            default='')
        parser.add_argument('--margin', required=False, type=float,
                            default=0.1)
        parser.add_argument('--surrogate_model_path1', required=True,
                            default='')
        parser.add_argument('--surrogate_model_path2', required=True,
                            default='')
        parser.add_argument('--surrogateUpdateFreq', required=False, type=float,
                            default=16)
        parser.add_argument('--iter_num', required=False, type=int,
                            default=5)
        parser.add_argument('--save_epoch_freq', type=int, 
                            default=1)
        parser.add_argument('--ink_1', required=False, type=float,
                            default=1)
        parser.add_argument('--ink_2', required=False, type=float,
                            default=0.5)
        parser.add_argument('--laser_1', required=False, type=float,
                            default=1)
        parser.add_argument('--laser_2', required=False, type=float,
                            default=0.5)        
        self.isTrain = True
        return parser

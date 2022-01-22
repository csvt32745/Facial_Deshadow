import configargparse

def config_parser_train():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--exp_name', type=str, default="")
    
    # for stacked hourglass network
    parser.add_argument('--out_gainbais', dest='is_out_gainbias', action='store_true')
    parser.add_argument('--n_stacks', type=int, default=1) 
    parser.add_argument('--recursive_stack', dest='is_recursive_stack', action='store_true')
    parser.add_argument('--ch_bottleneck', type=int, default=32) 

    
    # training parameters
    parser.add_argument('--add_shadow_weight', type=float, default=2)

    # for dataset processing
    # if not, using LAB; set to true if color_jitter is set
    parser.add_argument('--input_rgb', dest='is_rgb', action='store_true')
    # data augmentation
    # parser.add_argument('--color_jitter', dest='is_colorjitter', action='store_true')
    parser.add_argument('--color_jitter', type=str, default="mixed")
    parser.add_argument('--kernelratio_low', type=float, default=0.02)
    parser.add_argument('--kernelratio_high', type=float, default=0.05)
    parser.add_argument('--intensity_low', type=float, default=0.1)
    parser.add_argument('--intensity_high', type=float, default=0.7)

    # parser.add_argument('--which_model', type=str, default="TCSNET")
    parser.add_argument('--dataset_path', type=str, default="DPR_dataset/")
    parser.add_argument('--data_split_path', type=str, default="train_valid.list")
    parser.add_argument('--smaller_dataset', type=float, default=1)

    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--pretrain_epoch', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--save_model_path', type=str, default="./models/")
    parser.add_argument('--save_log_path', type=str, default="./log/")

    config = parser.parse_args()
    return config
import yaml
import argparse
import torch
import logging
import warnings
import time
import os
from intra_batch.utils import utils as utils
from trainer import Trainer

logger = logging.getLogger('GNNReID')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

warnings.filterwarnings("ignore")


def init_args():
    parser = argparse.ArgumentParser(description='Person Re-ID with GNN')
    parser.add_argument('--config_path', type=str, default='config/config_cub_train.yaml', help='Path to config file')
    parser.add_argument('--dataset_path', type=str, default='from_yaml', help='Give path to dataset, else path from yaml file will be taken')
    parser.add_argument('--bb_path', type=str, default='from_yaml', help='Give path to bb weight, else path from yaml file will be taken')
    parser.add_argument('--gnn_path', type=str, default='from_yaml', help='Give path to gnn weight, else path from yaml file will be taken')
    parser.add_argument('--net_type', type=str, default='from_yaml', help='Give net_type you want to use: resnet18/resnet32/resnet50/resnet101/resnet152/densenet121/densenet161/densenet169/densenet201/bn_inception')
    parser.add_argument('--is_apex', type=str, default='from_yaml', help='If you want to use apex set to 1')
    parser.add_argument('--device', type=str, default='0', help='can be set to 0 to num devices - 1')
    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.dataset_path != 'from_yaml':
        config['dataset']['dataset_path'] = args.dataset_path
    if args.bb_path != 'from_yaml':
        config['models']['encoder_params']['pretrained_path'] = args.bb_path
    if args.bb_path != 'from_yaml':
        config['models']['gnn_params']['pretrained_path'] = args.gnn_path
    if args.net_type != 'from_yaml':
        config['models']['encoder_params']['net_type'] = args.net_type
    if args.is_apex != 'from_yaml':
        config['train_params']['is_apex'] = args.is_apex

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    logger.info('Switching to device {}'.format(device))

    save_folder_results = config['eval_params']['result_save_dir'] #'results'
    save_folder_results = os.path.join(os.getcwd(), save_folder_results)
    utils.make_dir(save_folder_results)

    save_folder_nets = config['train_params']['model_save_dir']#'results_nets'
    save_folder_nets = os.path.join(os.getcwd(), save_folder_nets)
    utils.make_dir(save_folder_nets)

    trainer = Trainer(config, save_folder_nets, save_folder_results, device,
                      timer=time.time())
    trainer.train()


if __name__ == '__main__':
    args = init_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    main(args)

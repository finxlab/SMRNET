#!/usr/bin/env python
# coding: utf-8

from train import *

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger('best_hyperparameter.run')

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='SWMRNET', help='model type')
parser.add_argument('--model_name', type=str, default='model', help='model directory')
parser.add_argument('--loss', type=str, default='IC', help='loss type')
parser.add_argument('--random_seed', type=int, default=2026, help='random seed')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of data')
parser.add_argument('--test_date', type=str, default='2025-12-31', help='test date')
parser.add_argument('--setting', type=str, 
                    default='seed2026_lr0.0001_hd64_gh1_nh1_dr0.3', 
                    help='hyperparameter setting directory name')

                    
def test(configs, path):
    dataset_all = Dataset_all(configs)
    _, test_loader = dataset_all._get_data('test')
    
    cuda_exist = torch.cuda.is_available()
    torch.cuda.init()
    if cuda_exist:
        configs.device = torch.device('cuda')
        logger.info('Using Cuda...')
        model = UnifiedModel(configs).cuda()
    else:
        configs.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = UnifiedModel(configs)
    
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
    logger.info('Starting test evaluation...')
    summary_test = evaluate(model, test_loader, configs)
    
    return summary_test




if __name__ == '__main__':
    configs = parser.parse_args()
    set_seed(configs.random_seed)
    
    path = os.path.join(configs.model_name, configs.model_type,
                        configs.test_date, configs.setting)
    
    json_path = os.path.join(path, 'configs.json')
    configs_dict = load_json(json_path)
    for k, v in configs_dict.items():
        setattr(configs, k, v)
    
    cuda_exist = torch.cuda.is_available()
    configs.device = torch.device('cuda' if cuda_exist else 'cpu')
    
    logger.info(f'Loading model from: {path}')
    logger.info(f'Best val score: {configs_dict["best score"]}')
    
    summary_test = test(configs, path)
    
    test_json_path = os.path.join(path, 'best_test_metric.json')
    save_dict_to_json(summary_test, test_json_path)
    logger.info(f'Test results saved to: {test_json_path}')
    logger.info(f'Test summary: {summary_test}')
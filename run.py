
from train import *
# In[2]:

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)


logger = logging.getLogger('Model.run')

parser = argparse.ArgumentParser()

# basic config
parser.add_argument('--random_seed', type = int, default=2026, help='Random Seed num')

# data loader
parser.add_argument('--data', type = str, default='', help = 'dataset type')
parser.add_argument('--model_type', type = str, default='SMRNET', help='model type')
parser.add_argument('--model_name', type = str, default='model', help='Directory containing params.json')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')

parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

parser.add_argument('--test_date', type=str, default='2025-12-31', help='test start date')


# Common Model Define

parser.add_argument('--feature_dim', type=int, default=28, 
                   help='Input feature dimension: Number of stock features (e.g., price, volume, technical indicators)')
parser.add_argument('--market_dim', type=int, default=63, 
                   help='Market feature dimension: Number of market features')
parser.add_argument('--rel_dim', type=int, default=1, 
                   help='Stock relation dimension: Number of relation features')


# Model Hyperparameters
parser.add_argument('--hidden_dim', type=int, default=32, 
                   help='RNN hidden state dimension: Internal representation size for time series pattern learning. Also used as out_layer input')
parser.add_argument('--n_heads', type=int, default=1,
                    help='Number of heads for Temporal average pooling')
parser.add_argument('--GAT_heads', type=int, default=1,
                    help='Number of heads for Gate attention network')
parser.add_argument('--dropout', type=float, default=0.5, 
                   help='Dropout probability: Fraction of units to drop (0.0-1.0) for model overfitting prevention')





# Optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='IC', help='loss function')
parser.add_argument('--_lambda', type=float, default=1, help='lambda for other loss')


if __name__ == '__main__':

    configs = parser.parse_args()

    set_seed(configs.random_seed)
    configs.model_dir = os.path.join(configs.model_name, configs.model_type, configs.test_date)
   
    if not os.path.exists(configs.model_dir):
        os.makedirs(configs.model_dir)

    configs.data_path = configs.data + '.csv'


    logger.info('Loading complete.')

    cuda_exist = torch.cuda.is_available()
    
    # Hyperparameter set
    configs.setting = 'seed{}_lr{}_hd{}_gh{}_nh{}_dr{}'.format(
                        configs.random_seed,    # seed: Random seed for reproducibility 
                        configs.learning_rate,  # lr:   Step size for gradient updates 
                        configs.hidden_dim,     # hd:   Dimensionality of hidden representations 
                        configs.GAT_heads,      # gh:   Number of heads in Graph Attention
                        configs.n_heads,        # nh:   Number of heads in Multi-Head Attention
                        configs.dropout,        # dr:   Dropout rate to prevent overfitting 
                        )
    
    
    
    logger.info('Loading the datasets...')
    dataset_all = Dataset_all(configs)
    cuda_exist = torch.cuda.is_available()
    torch.cuda.init()

    ################################## MODEL INIT
    if cuda_exist:
        configs.device = torch.device('cuda')
        logger.info('Using Cuda...')
        model = UnifiedModel(configs).cuda()

    else:
        configs.device  = torch.device('cpu')
        logger.info('Not using cuda...')
        model = UnifiedModel(configs)
    ################################## MODEL INIT
    
    logger.info(configs)
    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.AdamW(model.parameters(), lr=configs.learning_rate,  weight_decay=1e-4)
    logger.info('Starting training for {} epoch(s)'.format(configs.train_epochs))
    train_and_evaluate(model,
                    dataset_all,
                    optimizer,
                    configs)
                            
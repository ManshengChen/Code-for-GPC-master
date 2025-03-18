import argparse

def arg_parse():
        parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
        parser.add_argument('--DS', dest='DS', help='Dataset')
        parser.add_argument('--local', dest='local', action='store_const', 
                const=True, default=False)
        parser.add_argument('--glob', dest='glob', action='store_const', 
                const=True, default=False)
        parser.add_argument('--prior', dest='prior', action='store_const', 
                const=True, default=False)
        parser.add_argument('--lr', dest='lr', type=float,
                help='Learning rate.')
        parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                help='Number of graph convolution layers before each pooling')
        parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                help='')
        parser.add_argument('--cluster_emb', dest='cluster_emb', type=int, default=10, help='')
        parser.add_argument('--d', dest='d', type=int, default=10, help='')
        parser.add_argument('--eta', dest='eta', type=int, default=2, help='')
        parser.add_argument('--clusters', dest='clusters', type=int, default=2, help='')    
        parser.add_argument('--preprocess', dest='preprocess', default=False) 
        parser.add_argument('--loss', dest='loss', default='kl')
        parser.add_argument('--seed', dest="seed", default=45, type=int, help='Seed to reproduce results')
        
        parser.add_argument('--tuning_type', type=str, default="gpf-plus", help='\'gpf\' for GPF and \'gpf-plus\' for GPF-plus in the paper')
        parser.add_argument('--pnum', type=int, default = 5, help='The number of independent basis for GPF-plus')
        parser.add_argument('--model_file', type=str, default = 'model.pth', help='File path to read the model (if there is any)')
        parser.add_argument('--is_p', type=bool, default=True, help='Whether to train the P vector, otherwise train the fixed model.')

        return parser.parse_args()


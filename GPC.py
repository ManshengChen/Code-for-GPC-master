# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from arguments import arg_parse
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator, Cluster
from torch_geometric.data import DataLoader
from aug import TUDataset
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from evaluate_embedding import cluster_acc
import graph_prompt as Prompt
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class GPC(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GPC, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    
    self.cluster_embedding = Cluster(args.hidden_dim * args.num_gc_layers, args.cluster_emb)

    self.cluster_layer = Parameter(torch.Tensor(args.cluster_emb, args.cluster_emb))
    torch.nn.init.xavier_uniform_(self.cluster_layer.data)
    
    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.alpha = 1.0

  def from_pretrained(self, model_file):
    state_dict = torch.load(model_file)
    state_dict['cluster_layer'] = self.cluster_layer
    for key in state_dict.keys():
        if 'encoder.convs.0.nn.0.weight' == key:
            problematic_weight = state_dict[key]
            mlp = nn.Linear(problematic_weight.shape[0], 32).to(device)
            adjusted_weight = mlp(problematic_weight.t().to(device)).t()
            state_dict[key] = adjusted_weight
    self.load_state_dict(state_dict)
    
  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def get_results(self, loader,prompt,is_p):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = []
    cluster = []
    y = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
            z, q, _, _ = self.forward(x, edge_index, batch, num_graphs,prompt,is_p)
            embedding.append(z.cpu().numpy())
            cluster.append(q.cpu().numpy())
            y.append(data.y.cpu().numpy())
    embedding = np.concatenate(embedding, 0)
    cluster = np.concatenate(cluster, 0)
    y = np.concatenate(y, 0)
    return embedding, cluster, y

  def get_p(self, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cluster = []
    with torch.no_grad():
        for data in loader:
            data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
            z, q, _, _ = self.forward(x, edge_index, batch, num_graphs)
            cluster.append(q.cpu().numpy())
    cluster = torch.from_numpy(np.concatenate(cluster, 0))
    p_distribution = target_distribution(cluster).to(device)
    return p_distribution


  def forward(self, x, edge_index, batch, num_graphs,prompt,is_p):
    if dataset.data.x is None or np.shape(dataset.data.x)[1] == 0:
        x = torch.ones(batch.shape[0],1).to(device)

    y, M = self.encoder(x, edge_index, batch,prompt,is_p)
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)
    
    # Clustering layer
    z = self.cluster_embedding(y)
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)   
    q = q.pow((self.alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return z, q,  g_enc, l_enc

def KMeans_TSNE(emb,data_name):
    emb = emb.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=n_cluster, n_init=100)
    y_pred = kmeans.fit_predict(emb)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(emb)
    for cluster_label in range(n_cluster):
        plt.scatter(X_tsne[y_pred == cluster_label, 0], X_tsne[y_pred == cluster_label, 1], label=f'Cluster {cluster_label}')
    plt.title('t-SNE Visualization with K-Means Clusters')
    plt.legend()
    plt.savefig(data_name+'tsne.png')
    plt.clf()

if __name__ == '__main__':
    import time
    start = time.time() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arg_parse()
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    accuracies = {'acc':[], 'nmi':[], 'ari':[], 'randomforest':[]}
    epochs = 14
    log_interval = 1
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    dataset = TUDataset(path, name=DS).shuffle()
    dataset_num_features = max(dataset.num_features, 1)
    dataset_num_features = 7
    n_cluster = 2
    
    if DS == 'PTC_MR':
        dataset_num_features = 18 # 35
        n_cluster = 2
    elif DS == 'PTC_MM':
        dataset_num_features = 20 # 33
        n_cluster = 2
    elif DS == 'MUTAG':
        dataset_num_features = 7 # 46
        n_cluster = 2
    elif DS == 'ENZYMES':
        dataset_num_features = 3 # 50
        n_cluster = 6
    elif DS == 'COX2':
        dataset_num_features = 35 # 18
        n_cluster = 2
    elif DS == 'NCI1':
        dataset_num_features = 37 # 16
    elif DS == 'NCI109':
        dataset_num_features = 38 # 15
        n_cluster = 2
    elif DS == 'COLLAB':
        dataset_num_features = 1 # 52
        n_cluster = 3
    dataloader = DataLoader(dataset, batch_size=batch_size)

    print('================')
    print('Dataset:', DS)
    print('args.seed:', args.seed)
    print('args.pnum:', args.pnum)
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('clutering embedding dimension: {}'.format(args.cluster_emb))
    print('================')

    iter = 1

    for it in range(iter):
        if args.tuning_type == 'gpf':
            prompt = Prompt.SimplePrompt(args.hidden_dim * args.num_gc_layers)   
        elif args.tuning_type == 'gpf-plus':
            prompt = Prompt.GPFplusAtt(args.hidden_dim * args.num_gc_layers, args.pnum)
        prompt.to(device)
        model = GPC(args.hidden_dim, args.num_gc_layers).to(device)
        if args.is_p :
            if not args.model_file == "":
                model.from_pretrained('save_model/BZR_'+DS+'_model.pth')
                for param in model.parameters():
                    param.requires_grad = False
            optimizer = torch.optim.Adam(prompt.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        mode='fd'
        measure='JSD'
        accmax = 0
        nmimax = 0
        arimax = 0

        loss_values = []
        for epoch in range(0, epochs):
            loss_all = 0
            batch = 0

            if epoch == 3:
                model.eval()
                emb, _, y = model.get_results(dataloader,prompt,args.is_p)
                print('kmeans_n_cluster',n_cluster)
                kmeans = KMeans(n_clusters=n_cluster, n_init=100)
                y_pred = kmeans.fit_predict(emb)
                print('kmeans.cluster_centers_.shape',kmeans.cluster_centers_.shape)
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
            model.train()

            for i, data in enumerate(dataloader):
                data = data.to(device)
                optimizer.zero_grad()
                _, q, g_enc, l_enc = model(data.x, data.edge_index, data.batch, data.num_graphs,prompt,args.is_p)
                if i == 0:
                    accumulated_g_enc = l_enc
                else:
                    accumulated_g_enc = torch.cat((accumulated_g_enc, l_enc), dim=0)
                KMeans_TSNE(g_enc,DS)
                local_global_loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)

                if epoch >= 3:
                    p = target_distribution(q)
                    kl_loss = F.kl_div(q.log(), p)
                    loss = local_global_loss + kl_loss
                    batch += 1  
                else:
                    loss = local_global_loss
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            loss_values.append(loss_all / len(dataloader))
            print('loss_loss',loss_all / len(dataloader))

            if epoch % log_interval == 0:
                model.eval()
                emb, q, y = model.get_results(dataloader,prompt,args.is_p)
                print('huatu_n_cluster',n_cluster)
                y_pred = q.argmax(1)
                acc = cluster_acc(y, y_pred)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                if acc > accmax:
                    accmax = acc
                if nmi > nmimax:
                    nmimax = nmi
                if ari > arimax:
                    arimax = ari

                print('===== Clustering performance: =====')
                print('Acc {:.4f}'.format(accmax), ', nmi {:.4f}'.format(nmimax), ', ari {:.4f}'.format(arimax))

    #    if not args.is_p:
    #        torch.save(model.state_dict(), 'save_model/'+DS+'_'+args.model_file)
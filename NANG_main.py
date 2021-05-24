from __future__ import print_function
import argparse
import torch
from tqdm import tqdm
import torch.utils.data
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
from NANG_models import LFI, Discriminator
from sklearn.utils import shuffle
import random
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from utils import load_data, accuracy, new_load_data, MMD, normalize_adj
import os
import pickle
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from evaluation import RECALL_NDCG
from sklearn.metrics import average_precision_score

# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
method_name = 'LFI'
train_fts_ratio = 0.4*1.0
train_link_ratio = 0.6*1.0


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='cora', help='cora, citeseer, steam, flickr')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--enc-name', type=str, default='GCN', help='Initial encoder model, GCN or GAT')
parser.add_argument('--alpha', type=float, default=0.2, help='Initial alpha for leak relu when use GAT as enc-name')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--neg_times', type=int, default=1, help='neg times with the positive pairs')
parser.add_argument('--n_gene', type=int, default=2, help='epoch number of generator')
parser.add_argument('--n_disc', type=int, default=1, help='epoch number of dsiscriminator')
parser.add_argument('--lambda_recon', type=float, default=1.0, help='lambda for reconstruction, always 1.0 in our model')
parser.add_argument('--lambda_cross', type=float, default=10.0, help='lambda for cross stream')
parser.add_argument('--lambda_gan', type=float, default=1.0, help='lambda for GAN loss, always 1.0 in our model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)
print('beging...............')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

graph_loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
BCE = torch.nn.BCEWithLogitsLoss(reduction='none')


if __name__ == "__main__":
    # Load data
    print('loading dataset: {}'.format(args.dataset))
    # note that the node_class_lbls, node_idx_train, node_idx_val, node_idx_test are only used for evaluation.
    adj, true_features, node_class_lbls, _, _, _ = new_load_data(args.dataset, norm_adj=False, generative_flag=True)

    # pickle.dump(node_class_lbls.numpy(), open(os.path.join(os.getcwd(), 'data', args.dataset, '{}_labels.pkl'.format(args.dataset)), 'wb'))
    # pickle.dump(adj.to_dense().sum(1), open(os.path.join(os.getcwd(),
    #                                                      'features', method_name, '{}_degree.pkl'.format(args.dataset)), 'wb'))
    norm_adj, _, _, _, _, _ = new_load_data(args.dataset, norm_adj=True, generative_flag=True)
    norm_adj_arr = norm_adj.to_dense().numpy()

    # generate ont-hot features for all nodes, this means no node feature is used
    indices = torch.LongTensor(np.stack([np.arange(adj.shape[0]), np.arange(adj.shape[0])], axis=0))
    values = torch.FloatTensor(np.ones(indices.shape[1]))
    diag_fts = torch.sparse.FloatTensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]]))

    # split train features and generative features
    shuffled_nodes = shuffle(np.arange(adj.shape[0]), random_state=args.seed)
    train_idx = torch.LongTensor(shuffled_nodes[:int(train_fts_ratio * adj.shape[0])])
    vali_idx = torch.LongTensor(
        shuffled_nodes[int(0.4 * adj.shape[0]):int((0.4 + 0.1) * adj.shape[0])])
    test_idx = torch.LongTensor(shuffled_nodes[int((0.4 + 0.1) * adj.shape[0]):])

    pickle.dump(train_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_train_idx.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))
    pickle.dump(vali_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_vali_idx.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))
    pickle.dump(test_idx, open(os.path.join(os.getcwd(), 'features', method_name, '{}_{}_test_idx.pkl'.format(
        args.dataset, train_fts_ratio)), 'wb'))

    # set loss function and pos weight
    if args.dataset in ['cora', 'citeseer', 'steam', 'amazon_computer', 'amazon_photo']:
        pos_weight = torch.sum(true_features == 0.0) / (
            torch.sum(true_features != 0.0))
        fts_loss_func = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    elif args.dataset in ['pubmed', 'coauthor_cs']:
        fts_loss_func = torch.nn.MSELoss(reduction='none')

    '''
    make data preparation
    '''
    # get train data for A
    if not os.path.exists(os.path.join(os.getcwd(), 'data', args.dataset, 'train_data_{}.pkl'.format(train_link_ratio))):
        all_pos_indices = np.where(adj.to_dense().numpy() != 0)
        all_pos_indices = shuffle(np.array(list(zip(all_pos_indices[0], all_pos_indices[1]))),
                                  random_state=args.seed)

        train_pos_indices = all_pos_indices[:int(train_link_ratio * len(all_pos_indices))]
        val_pos_indices = all_pos_indices[int(0.6 * len(all_pos_indices)):int(0.8 * len(all_pos_indices))]
        test_pos_indices = all_pos_indices[int(0.8 * len(all_pos_indices)):]

        all_neg_indices = np.where(norm_adj_arr == 0)
        all_neg_indices = np.array(list(zip(all_neg_indices[0], all_neg_indices[1])))
        all_neg_indices = shuffle(all_neg_indices, random_state=args.seed)[:len(all_pos_indices)]

        train_neg_indices = all_neg_indices[:int(train_link_ratio * len(all_neg_indices))]
        val_neg_indices = all_neg_indices[int(0.6 * len(all_neg_indices)):int(0.8 * len(all_neg_indices))]
        test_neg_indices = all_neg_indices[int(0.8 * len(all_neg_indices)):]

        train_data = {}
        val_data = {}
        test_data = {}
        train_data['indices'] = np.concatenate([train_pos_indices,
                                                train_neg_indices], axis=0)
        train_data['labels'] = np.concatenate([np.ones(len(train_pos_indices)),
                                               np.zeros(len(train_neg_indices))])
        val_data['indices'] = np.concatenate([val_pos_indices,
                                                val_neg_indices], axis=0)
        val_data['labels'] = np.concatenate([np.ones(len(val_pos_indices)),
                                               np.zeros(len(val_neg_indices))])
        test_data['indices'] = np.concatenate([test_pos_indices,
                                              test_neg_indices], axis=0)
        test_data['labels'] = np.concatenate([np.ones(len(test_pos_indices)),
                                             np.zeros(len(test_neg_indices))])

        pickle.dump(train_data, open(os.path.join(os.getcwd(), 'data', args.dataset, 'train_data_{}.pkl'.format(train_link_ratio)), 'wb'))
        pickle.dump(val_data, open(os.path.join(os.getcwd(), 'data', args.dataset, 'val_data_{}.pkl'.format(train_link_ratio)), 'wb'))
        pickle.dump(test_data, open(os.path.join(os.getcwd(), 'data', args.dataset, 'test_data_{}.pkl'.format(train_link_ratio)), 'wb'))

    else:
        train_data = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset, 'train_data_{}.pkl'.format(train_link_ratio)), 'rb'))
        val_data = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset, 'val_data_{}.pkl'.format(train_link_ratio)), 'rb'))
        test_data = pickle.load(open(os.path.join(os.getcwd(), 'data', args.dataset, 'test_data_{}.pkl'.format(train_link_ratio)), 'rb'))

    # get A for X2A in training process
    df = pd.DataFrame(data={'rows': train_data['indices'][:, 0], 'cols': train_data['indices'][:, 1],
                                'labels': train_data['labels']}, columns=['rows', 'cols', 'labels'])
    df = df[df['rows'].isin(train_idx) & df['cols'].isin(train_idx)]
    train_X2A_data = {}
    train_X2A_data['indices'] = df.values[:, :-1]
    train_X2A_data['labels'] = df.values[:, -1]

    train_adj = np.zeros(adj.shape)
    indices = train_data['indices'][np.where(train_data['labels'] != 0)]
    train_adj[indices[:, 0], indices[:, 1]] = 1.0
    # train_adj = adj.to_dense().numpy()
    # train_adj[val_data['indices'][:, 0], val_data['indices'][:, 1]] = 0.0
    # train_adj[test_data['indices'][:, 0], test_data['indices'][:, 1]] = 0.0
    train_norm_adj = normalize_adj(sp.coo_matrix(train_adj) + sp.eye(train_adj.shape[0]))

    indices = torch.LongTensor(np.stack([train_norm_adj.tocoo().row, train_norm_adj.tocoo().col], axis=0))
    values = torch.FloatTensor(train_norm_adj.tocoo().data)
    train_norm_adj = torch.sparse.FloatTensor(indices, values, torch.Size(train_norm_adj.shape))


    if args.cuda:
        train_norm_adj = train_norm_adj.cuda()
        diag_fts = diag_fts.cuda()
        true_features = true_features.cuda()

    '''
    define things for LFI (i.e. SAT) model
    '''
    prior = torch.distributions.normal.Normal(loc=torch.FloatTensor([0.0]), scale=torch.FloatTensor([1.0]))
    model = LFI(n_nodes=train_norm_adj.shape[0], n_fts=true_features.shape[1], n_hid=args.hidden, dropout=args.dropout, args=args)
    disc = Discriminator(args.hidden, args.hidden, args.dropout)
    if args.cuda:
        model.cuda()
        disc.cuda()

    g_optimizer = optim.Adam(model.parameters(), lr=1e-3,
                             weight_decay=args.weight_decay)
    d_optimizer = optim.SGD(disc.parameters(), lr=1e-3,
                             weight_decay=args.weight_decay)


    train_G_loss_list = []
    train_D_loss_list = []
    train_recon_loss_list = []
    auc_values_list = []
    f1_values_list = []

    # # set params to calculate MMD distance
    # sigma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4]
    # sigma_list = torch.FloatTensor(np.array(sigma_list))
    # if args.cuda:
    #     sigma_list = sigma_list.cuda()

    best = 0.0
    # best_mse = 10000.0
    bad_counter = 0
    best_epoch = 0

    if train_norm_adj.is_sparse and args.enc_name=='GAT':
        train_norm_adj = train_norm_adj.to_dense()
    for epoch in range(1, args.epochs + 1):
        model.train()
        '''
        train the generators
        '''

        for i in range(1, args.n_gene + 1):
            # train model
            g_optimizer.zero_grad()

            ae_z, ae_fts, ae_adj_z, gae_z, gae_fts, gae_adj_z = model(true_features, train_norm_adj, diag_fts)

            ae_adj_preds = (ae_adj_z[train_X2A_data['indices'][:,0]]*ae_adj_z[train_X2A_data['indices'][:,1]]).sum(1)
            ae_adj_labels = torch.FloatTensor(train_X2A_data['labels'])

            gae_adj_preds = (gae_adj_z[train_data['indices'][:,0]]*gae_adj_z[train_data['indices'][:,1]]).sum(1)
            gae_adj_labels = torch.FloatTensor(train_data['labels'])
            if args.cuda:
                ae_adj_labels = ae_adj_labels.cuda()
                gae_adj_labels = gae_adj_labels.cuda()

            X_loss = args.lambda_recon*fts_loss_func(ae_fts[train_idx], true_features[train_idx]).sum(1).mean()

            A2X_loss = args.lambda_cross*fts_loss_func(gae_fts[train_idx], true_features[train_idx]).sum(1).mean()
            X2A_loss = args.lambda_cross*graph_loss_func(ae_adj_preds, ae_adj_labels).mean()
            A_loss = args.lambda_recon*graph_loss_func(gae_adj_preds, gae_adj_labels).mean()

            fake_logits_ae = disc(ae_z).reshape([-1])
            fake_logits_gae = disc(gae_z).reshape([-1])

            G_lbls_1 = torch.ones_like(fake_logits_ae)

            G_loss_ae = BCE(fake_logits_ae, G_lbls_1).mean()
            G_loss_gae = BCE(fake_logits_gae, G_lbls_1).mean()

            G_loss = args.lambda_gan*(G_loss_ae + G_loss_gae)

            recon_loss = X_loss + X2A_loss + A2X_loss + A_loss

            (G_loss+recon_loss).backward()
            g_optimizer.step()


        '''
        train the discriminator
        '''

        for i in range(1, args.n_disc + 1):
            # train model
            d_optimizer.zero_grad()
            ae_z, ae_fts, ae_adj_z, gae_z, gae_fts, gae_adj_z = model(true_features, train_norm_adj, diag_fts)
            ae_z = ae_z[train_idx]
            # gae_z = gae_z
            # Sample noise as discriminator ground truth
            # standard Gaussian
            true_z_ae = prior.sample([ae_z.shape[0], ae_z.shape[1]]).reshape([ae_z.shape[0], ae_z.shape[1]])
            true_z_gae = prior.sample([gae_z.shape[0], gae_z.shape[1]]).reshape([gae_z.shape[0], gae_z.shape[1]])

            if args.cuda:
                true_z_ae = true_z_ae.cuda()
                true_z_gae = true_z_gae.cuda()
            true_logits_ae = disc(true_z_ae).reshape([-1])
            true_logits_gae = disc(true_z_gae).reshape([-1])
            fake_logits_ae = disc(ae_z).reshape([-1])
            fake_logits_gae = disc(gae_z).reshape([-1])

            logits_ae = torch.cat([true_logits_ae, fake_logits_ae])
            logits_gae = torch.cat([true_logits_gae, fake_logits_gae])

            D_lbls_ae = torch.cat([torch.ones_like(true_logits_ae), torch.zeros_like(fake_logits_ae)])
            D_lbls_gae = torch.cat([torch.ones_like(true_logits_gae), torch.zeros_like(fake_logits_gae)])

            D_loss_ae = BCE(logits_ae, D_lbls_ae).mean()
            D_loss_gae = BCE(logits_gae, D_lbls_gae).mean()

            D_loss = args.lambda_gan*(D_loss_ae + D_loss_gae)

            D_loss.backward()
            d_optimizer.step()

        train_D_loss_list.append(D_loss.item())
        train_G_loss_list.append(G_loss.item())
        train_recon_loss_list.append(recon_loss.item())

        # make evaluation process
        model.eval()

        ae_z, ae_fts, ae_adj_z, gae_z, gae_fts, gae_adj_z = model(true_features, train_norm_adj, diag_fts)

        gae_adj_preds = (gae_adj_z[val_data['indices'][:, 0]] * gae_adj_z[val_data['indices'][:, 1]]).sum(1)
        gae_adj_preds = torch.sigmoid(gae_adj_preds)

        if args.cuda:
            gae_adj_preds = gae_adj_preds.data.cpu().numpy()
        else:
            gae_adj_preds = gae_adj_preds.data.numpy()

        auc = roc_auc_score(val_data['labels'], gae_adj_preds)
        # f1 = f1_score(val_data['labels'], (gae_adj_preds>0.5).astype(np.int32))
        auc_values_list.append(auc)
        # f1_values_list.append(f1)
        if auc_values_list[-1] > best:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'output', method_name,
                                                        'best_LFI_{}_{}_{}_G{}_R{}_C{}.pkl'.format(args.dataset,
                                                                                                train_fts_ratio,
                                                                                                train_link_ratio,
                                                                                                args.lambda_gan,
                                                                                                args.lambda_recon,
                                                                                                args.lambda_cross)))
            best = auc_values_list[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1


        '''
        print training and validation information
        '''
        if epoch % 1 == 0:
            print('Train Epoch: {}, X loss: {:.8f}, A2X loss: {:.8f}, X2A loss: {:.8f}, '
                  'A loss: {:.8f}, G loss: {:.8f}, D loss: {:.8f}, val auc:{:.4f}'.format(
                    epoch, X_loss.item(), A2X_loss.item(), X2A_loss.item(), A_loss.item(),
                    G_loss.item(), D_loss.item(), auc))

print("LFI Optimization Finished!")
print("Train fts ratio: {}, best epoch: {}".format(train_fts_ratio, best_epoch))

print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'output', method_name,
                                              'best_LFI_{}_{}_{}_G{}_R{}_C{}.pkl'.format(args.dataset,
                                                                                      train_fts_ratio,
                                                                                      train_link_ratio,
                                                                                      args.lambda_gan,
                                                                                      args.lambda_recon,
                                                                                      args.lambda_cross))))

'''
evaluation
'''
# find neighbors and make raw feature aggregation for unknown nodes
model.eval()
ae_z, ae_fts, ae_adj_z, gae_z, gae_fts, gae_adj_z = model(true_features, train_norm_adj, diag_fts)
gae_adj_preds = (gae_adj_z[test_data['indices'][:, 0]] * gae_adj_z[test_data['indices'][:, 1]]).sum(1)
gae_adj_preds = torch.sigmoid(gae_adj_preds)

if args.cuda:
    gae_adj_preds = gae_adj_preds.data.cpu().numpy()
else:
    gae_adj_preds = gae_adj_preds.data.numpy()

auc = roc_auc_score(test_data['labels'], gae_adj_preds)
ap = average_precision_score(test_data['labels'], gae_adj_preds)
print('test over, auc:{:.4f}, average precision: {:.4f},'.format(auc, ap))


print('method: {}, dataset: {}, lambda GAN: {}, lambda cross: {}, hidden: {}'.format(method_name, args.dataset,
                                                                                     args.lambda_gan,
                                                                                     args.lambda_cross,
                                                                                     args.hidden))

print('method: {}, dataset: {}, kernel:{}, fts ratio: {}, link ratio: {}'.format(method_name, args.dataset, args.enc_name,
                                                                                 train_fts_ratio, train_link_ratio))

import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging

class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        
        return self.mlp(out)
    
class GATe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, n_heads=4, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        # GAT specific code
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True, edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
                
        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
            
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x

        return self.mlp(out)
    
class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x
        return self.mlp(out)
    
class RGCN(nn.Module):
    def __init__(self, num_features, edge_dim, num_relations, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_update=False,
                residual=True,
                dropout=0.0, final_dropout=0.5, n_bases=-1):
        super(RGCN, self).__init__()

        self.num_features = num_features
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.residual = residual
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.n_classes = n_classes
        self.edge_update = edge_update
        self.num_relations = num_relations
        self.n_bases = n_bases

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.ModuleList()

        if self.edge_update:
            self.emlps = nn.ModuleList()
            self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
        
        for _ in range(self.num_gnn_layers):
            conv = RGCNConv(self.n_hidden, self.n_hidden, num_relations, num_bases=self.n_bases)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

            if self.edge_update:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, RGCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_type = edge_attr[:, -1].long()
        #edge_attr = edge_attr[:, :-1]
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x =  (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_type)))) / 2
            if self.edge_update:
                edge_attr = (edge_attr + F.relu(self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)))) / 2
        
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        x = self.mlp(x)
        out = x

        return x
    

class GraphAutoEncoder(torch.nn.Module):

    def __init__(self, num_features, num_gnn_layers=2, n_classes=2, 
                n_hidden=64, edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5, latent_dim=32):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
            ), edge_dim=self.n_hidden)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.encoder_final = nn.Linear(n_hidden, latent_dim)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, edge_dim)
        )


        self.mlp = nn.Sequential(
            Linear(latent_dim*2 + edge_dim + 1, 50),
            nn.ReLU(), 
            nn.Dropout(self.final_dropout),
            Linear(50, 25), 
            nn.ReLU(), 
            nn.Dropout(self.final_dropout),
            Linear(25, n_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        orig_edge_attr = edge_attr.clone()

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2

        z = self.encoder_final(x)
        
        src, dst = edge_index

        z_src = z[src]
        z_dst = z[dst]

        edge_input = torch.cat([z_src, z_dst], dim=1)
        reconstructed_edge_attr = self.decoder_mlp(edge_input)

        recon_error = F.mse_loss(reconstructed_edge_attr, orig_edge_attr, reduction='none').mean(dim=1, keepdim=True)

        z_features = z[edge_index.T].reshape(-1, 2 * self.latent_dim)
        features = torch.cat((z_features, orig_edge_attr, recon_error), 1)

        out = self.mlp(features)

        return out

class MultiPNAAutoEncoder(torch.nn.Module):
    def __init__(self, num_features, edge_dim, num_gnn_layers=2, 
                 n_hidden=20, latent_dim=12, edge_updates=False, 
                 use_reverse_mp=False, use_ports=False, use_ego_ids=False, 
                 dropout=0.08, final_dropout=0.29, n_classes=2, deg=None):
        self.e_dim = edge_dim
        super().__init__()
        
        n_hidden = int((n_hidden // 5) * 5)
        
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.use_reverse_mp = use_reverse_mp
        self.use_ports = use_ports
        self.use_ego_ids = use_ego_ids
        self.dropout = dropout
        self.final_dropout = final_dropout
        
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        self.fwd_convs = nn.ModuleList()
        self.fwd_bns = nn.ModuleList()
        
        for _ in range(num_gnn_layers):
            conv = PNAConv(
                in_channels=n_hidden, 
                out_channels=n_hidden,
                aggregators=self.aggregators, 
                scalers=self.scalers, 
                deg=deg,
                edge_dim=n_hidden, 
                towers=5, 
                pre_layers=1, 
                post_layers=1,
                divide_input=False
            )
            self.fwd_convs.append(conv)
            self.fwd_bns.append(BatchNorm(n_hidden))
        
        if use_reverse_mp:
            self.rev_convs = nn.ModuleList()
            self.rev_bns = nn.ModuleList()
            
            for _ in range(num_gnn_layers):
                conv = PNAConv(
                    in_channels=n_hidden, 
                    out_channels=n_hidden,
                    aggregators=self.aggregators, 
                    scalers=self.scalers, 
                    deg=deg,
                    edge_dim=n_hidden, 
                    towers=5, 
                    pre_layers=1, 
                    post_layers=1,
                    divide_input=False
                )
                self.rev_convs.append(conv)
                self.rev_bns.append(BatchNorm(n_hidden))
        
        if edge_updates:
            self.edge_mlps = nn.ModuleList()
            
            for _ in range(num_gnn_layers):
                self.edge_mlps.append(nn.Sequential(
                    nn.Linear(3 * n_hidden, n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden)
                ))
        
        self.encoder_final = nn.Linear(n_hidden, latent_dim)
        
        decoder_input_dim = 2 * latent_dim
        

        self.decoder_expand = nn.Sequential(
            nn.Linear(decoder_input_dim, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(n_hidden),
                nn.Dropout(dropout)
            ))
            
        self.decoder_final = nn.Linear(n_hidden, edge_dim)
        

        classifier_input_dim = 2 * latent_dim + edge_dim + 1 
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 50), 
            nn.ReLU(), 
            nn.Dropout(final_dropout),
            nn.Linear(50, 25), 
            nn.ReLU(), 
            nn.Dropout(final_dropout),
            nn.Linear(25, n_classes)
        )

    def forward(self, x, edge_index, edge_attr):

        if isinstance(edge_index, dict):
            src, dst = edge_index[('node', 'to', 'node')]
            edge_attr_forward = edge_attr[('node', 'to', 'node')]
            node_x = x['node']
            
            if self.use_reverse_mp:
                src_rev, dst_rev = edge_index[('node', 'rev_to', 'node')]
                edge_attr_rev = edge_attr[('node', 'rev_to', 'node')]
        else:
            src, dst = edge_index
            edge_attr_forward = edge_attr
            node_x = x
            
            if self.use_reverse_mp:
                src_rev, dst_rev = edge_index.flip(0)
                edge_attr_rev = edge_attr 
                
        if edge_attr_forward.shape[1] > self.e_dim:
            edge_attr_forward = edge_attr_forward[:, 1:]
            if self.use_reverse_mp:
                edge_attr_rev = edge_attr_rev[:, 1:]
        
        orig_edge_attr = edge_attr_forward.clone()
        
        node_x = self.node_emb(node_x)
        edge_attr_forward = self.edge_emb(edge_attr_forward)
        
        if self.use_reverse_mp:
            edge_attr_rev = self.edge_emb(edge_attr_rev)
        
        for i in range(self.num_gnn_layers):
            if isinstance(edge_index, dict):
                forward_index = edge_index[('node', 'to', 'node')]
            else:
                forward_index = edge_index
                
            node_x_fwd = F.relu(self.fwd_bns[i](self.fwd_convs[i](node_x, forward_index, edge_attr_forward)))
            
            if self.use_reverse_mp:
                if isinstance(edge_index, dict):
                    reverse_index = edge_index[('node', 'rev_to', 'node')]
                else:
                    reverse_index = edge_index.flip(0)
                    
                node_x_rev = F.relu(self.rev_bns[i](self.rev_convs[i](node_x, reverse_index, edge_attr_rev)))

                node_x = (node_x + node_x_fwd + node_x_rev) / 3
            else:
                node_x = (node_x + node_x_fwd) / 2
                
            if self.edge_updates:
                edge_attr_forward = edge_attr_forward + self.edge_mlps[i](torch.cat([node_x[src], node_x[dst], edge_attr_forward], dim=-1)) / 2
                
                if self.use_reverse_mp:
                    edge_attr_rev = edge_attr_rev + self.edge_mlps[i](torch.cat([node_x[src_rev], node_x[dst_rev], edge_attr_rev], dim=-1)) / 2
        

        z = self.encoder_final(node_x)
        
        z_src = z[src]
        z_dst = z[dst]
        edge_input = torch.cat([z_src, z_dst], dim=1)
            
        h = self.decoder_expand(edge_input)
        
        for i in range(self.num_gnn_layers):
            h = self.decoder_layers[i](h)
        
        reconstructed_edge_attr = self.decoder_final(h)
        
        recon_error = F.mse_loss(reconstructed_edge_attr, orig_edge_attr, reduction='none').mean(dim=1, keepdim=True)
        
        edge_features = torch.cat([z[src], z[dst], orig_edge_attr, recon_error], dim=1)
        out = self.classifier(edge_features)
        
        if isinstance(edge_index, dict):
            return {('node', 'to', 'node'): out}
        else:
            return out

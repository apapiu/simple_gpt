import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from itertools import cycle
import numpy as np
from tqdm import tqdm
import lightning as L
from pytorch_lightning.loggers import WandbLogger


class SequenceGenerator(IterableDataset):
    def __init__(self, token_ids, seq_length, batch_size):
        self.token_ids = torch.tensor(token_ids)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.n_tokens = len(token_ids)
        self.indices = torch.arange(0, self.n_tokens - seq_length)

    def __iter__(self):
        self.indices = self.indices[torch.randperm(len(self.indices))]
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            X_batch = self.token_ids[batch_indices[:, None] + torch.arange(self.seq_length)]
            y_batch = self.token_ids[batch_indices[:, None] + torch.arange(1, self.seq_length + 1)]
            yield X_batch, y_batch

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, mlp_multiplier):
        super(AttentionBlock, self).__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout))

    def forward(self, x):
        q, k, v = self.q_linear(x), self.k_linear(x), self.v_linear(x)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        x = x + self.dropout(attn)
        x = x + self.dropout(self.mlp(x))
        return x

class Transformer(L.LightningModule):
    def __init__(self, vocab_size, embed_dim, seq_length, n_heads, attention_layers, dropout, mlp_multiplier, lr, epsilon):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_length, embed_dim)
        self.attention_blocks = nn.ModuleList([AttentionBlock(embed_dim, n_heads, dropout, mlp_multiplier) for _ in range(attention_layers)])
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=epsilon)
        self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_length).long())

    def forward(self, x):

        pos_enc = self.precomputed_pos_enc.expand(x.size(0), -1)
        x = self.embedding(x) + self.pos_embedding(pos_enc)
        for block in self.attention_blocks:
            x = block(x)
        x = F.gelu(self.fc(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        wandb.log({"train_loss": loss}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        wandb.log({"val_loss": loss}, step=self.global_step)
        return loss

    def configure_optimizers(self):
        return self.optimizer

if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 2000
    embed_dim = 256
    seq_length = 128
    n_heads = 4
    attention_layers = 4
    dropout = 0.2
    mlp_multiplier = 2
    lr = 1e-3
    epsilon = 1e-7
    
    max_steps = 10000
    val_check_interval=1000
    
    # Data
    token_ids = np.array(train_ids)
    batch_size = 512
    wandb_logger = WandbLogger()
    
    wandb.init(
        project="simplebooks_gpt",
        config = {
        'seq_length': seq_length,
        'embed_dim': embed_dim,
        #'use_positional_emb': use_positional_emb,
        'attention_layers': attention_layers,
        'n_heads': n_heads,
        'lr': lr,
        'epsilon':epsilon,
        'dropout': dropout,
        #'lr_decay': lr_decay,
        'batch_size': batch_size,
        #'use_layer_norm':use_layer_norm,
        'mlp_multiplier':mlp_multiplier,
        'train_data_url':train_data_url,
        'val_data_url':val_data_url,
        'vocab_size':vocab_size,
        'ntrain':len(train_ids),
        'nval':len(val_ids)
    })
    
    
    # Initialize
    train_gen = SequenceGenerator(token_ids, seq_length, batch_size)
    train_loader = DataLoader(train_gen, batch_size=None, num_workers=1)
    val_gen = SequenceGenerator(val_ids, seq_length, batch_size)
    val_loader = DataLoader(val_gen, batch_size=None, num_workers=1)
    
    model = Transformer(vocab_size, embed_dim, seq_length, n_heads, attention_layers, dropout, mlp_multiplier, lr, epsilon)
    
    trainer = L.Trainer(max_steps=max_steps,
                        val_check_interval=val_check_interval,
                        logger=wandb_logger,
                        precision=16,
                        limit_train_batches=len(train_ids)//batch_size)
    
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

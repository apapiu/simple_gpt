#!pip install lightning
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.callbacks import ModelCheckpoint
from IPython.display import clear_output

def sample_top_k(probs, k):

    sorted_indices = np.argsort(probs)
    top_k_indices = sorted_indices[-k:]
    top_k_probs = probs[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    sampled_index = np.random.choice(top_k_indices, p=top_k_probs)
    return sampled_index

# def generate_text(example, model, nchar=128, k=5,
#                   one_char_at_a_time=False, end_on_zero=True, device="cuda"):
                      
#     model.eval()  # Set the model to evaluation mode

#     with torch.no_grad():  # Disable gradient calculations for inference
#         for i in range(nchar):
#             if one_char_at_a_time:
#                 clear_output(wait=True)

#             from torch.nn.functional import pad

#             if len(example) < seq_length:
#                 padding_length = seq_length - len(example)
#                 example_torch = torch.tensor(example).long()
#                 example_torch = pad(example_torch, (padding_length, 0), 'constant', 0).unsqueeze(0).to(device)
#             else:
#                 example_torch = torch.tensor(example[-seq_length:]).unsqueeze(0).long().to(device)

#             # Forward pass
#             logits = model(example_torch)
#             probs = torch.nn.functional.softmax(logits[0][-1], dim=0).cpu().numpy()

#             next_token = sample_top_k(probs, k)

#             example.append(next_token)

#             if one_char_at_a_time:
#                 print(decode(example).replace('??', ' '))

#             if next_token == 0 and end_on_zero:
#                 break

#         if not one_char_at_a_time:
#             print(decode(example).replace('??', ' '))

#     return decode(example).replace('??', ' ')

def generate_text(example, model, nchar=128, k=5,
                  one_char_at_a_time=False, end_on_zero=True, device="cuda"):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations for inference
        for i in range(nchar):

            if one_char_at_a_time:
                clear_output(wait=True)

            example_torch = torch.tensor(example[-seq_length:]).unsqueeze(0).long().to(device)

            logits = model(example_torch) #1, n, vocab -> pick

            probs = torch.nn.functional.softmax(logits[0][-1], dim=0).cpu().numpy()
            next_token = sample_top_k(probs, k)

            example.append(next_token)

            if one_char_at_a_time:
                print(decode(example))

            if next_token == 0 and end_on_zero:
                break

        if not one_char_at_a_time:
            print(decode(example))

    return decode(example)




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
        self.n_heads = n_heads
        self.dropout_level = dropout
        self.dropout = nn.Dropout(self.dropout_level)
        self.d_k = embed_dim // n_heads
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.ReLU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
        )

    def forward(self, x):
        q, k, v = self.q_linear(x), self.k_linear(x), self.v_linear(x)

        d_k = self.d_k
        #split into heads -> (bs, h, n, d_k)
        q, k, v = [x.view(x.size(0), x.size(1), self.n_heads, d_k).permute(0, 2, 1, 3) for x in [q, k, v]]

        #TODO: use sliding window attention here?
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                is_causal=True, dropout_p=self.dropout_level)
        attn =  attn.permute(0, 2, 1, 3).contiguous().view(attn.size(0), attn.size(2), -1)

        x = x + self.dropout(attn)
        x = x + self.dropout(self.mlp(x))
        return x

class Transformer(L.LightningModule):
    ### more docs: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    def __init__(self, vocab_size, embed_dim, seq_length, n_heads, attention_layers, dropout, mlp_multiplier, lr, epsilon, max_steps):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_length, embed_dim)
        self.attention_blocks = nn.ModuleList([AttentionBlock(embed_dim, n_heads, dropout, mlp_multiplier) for _ in range(attention_layers)])
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_steps = max_steps
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=epsilon)
        self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_length).long())

        self.batch_val_losses = []

    def forward(self, x):

        pos_enc = self.precomputed_pos_enc.expand(x.size(0), -1)
        x = self.embedding(x) + self.pos_embedding(pos_enc)
        for block in self.attention_blocks:
            x = block(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))

        if self.global_step % save_every_n_iterations == 0 and self.global_step>0:
            print('saving_model')
            checkpoint_path = f"model_checkpoint_{self.global_step}.pth"
            torch.save(self.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

        wandb.log({"train_loss": loss}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        self.batch_val_losses.append(loss.item())
        return loss

    def on_validation_epoch_end(self):

        val_loss = np.array(self.batch_val_losses).mean()
        self.batch_val_losses = []

        wandb.log({"val_loss": val_loss}, step=self.global_step) 
        wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']}, step=self.global_step)

        query = """To be or not to be ... """
        example = encode(query)
        gen = generate_text(example, model, nchar=128, k=3, one_char_at_a_time=False,  end_on_zero=False)
        #text_table.add_data(self.global_step, gen)

    def configure_optimizers(self):

        scheduler = {
            'scheduler': CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=model.optimizer.param_groups[0]['lr']/10),
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }
        return {'optimizer': self.optimizer, 'lr_scheduler': scheduler}

if __name__ == '__main__':
    # Hyperparameters
    vocab_size = len(mapping)
    embed_dim = 256
    seq_length = 256
    n_heads = 4
    attention_layers = 5
    dropout = 0.2
    mlp_multiplier = 2
    lr = 1e-3
    epsilon = 1e-7
    
    max_steps=51000
    val_check_interval=1000
    save_every_n_iterations = 10000
    
    # Data
    token_ids = np.array(train_ids).astype("int64")
    val_ids = np.array(val_ids).astype("int64")
    batch_size = 512
    wandb_logger = WandbLogger()
    
    wandb.init(
        project="nielsen_gpt",
        config = {
        'seq_length': seq_length,
        'embed_dim': embed_dim,
        'attention_layers': attention_layers,
        'n_heads': n_heads,
        'lr': lr,
        'epsilon':epsilon,
        'dropout': dropout,
        'batch_size': batch_size,
        'mlp_multiplier':mlp_multiplier,
        'vocab_size':vocab_size,
        'ntrain':len(train_ids),
        'nval':len(val_ids)
    })
    
    
    # Initialize
    train_gen = SequenceGenerator(token_ids, seq_length, batch_size)
    train_loader = DataLoader(train_gen, batch_size=None, num_workers=1)
    val_gen = SequenceGenerator(val_ids, seq_length, batch_size)
    val_loader = DataLoader(val_gen, batch_size=None, num_workers=1)
    
    model = Transformer(vocab_size, embed_dim, seq_length, n_heads, attention_layers, dropout, mlp_multiplier, lr, epsilon, max_steps)
    
    trainer = L.Trainer(max_steps=max_steps,
                        val_check_interval=val_check_interval,
                        logger=wandb_logger,
                        precision=16,
                        limit_train_batches=len(train_ids)//batch_size)
    
    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

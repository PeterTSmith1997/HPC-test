import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import nltk
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Download Brown corpus if not already downloaded
nltk.download('brown')
from nltk.corpus import brown

# --- Data Preparation using Brown Corpus ---
# Get sentences from the Brown corpus and filter for a useful length
raw_sentences = [' '.join(sent) for sent in brown.sents() if 3 <= len(sent) <= 10]
# Use a subset for faster training (e.g., 1000 sentences)
sentences = raw_sentences[:1000]
print(f"Using {len(sentences)} sentences from the Brown corpus.")

# Build vocabulary (assign index 0 for <PAD>)
words = set()
for sentence in sentences:
    for word in sentence.split():
        words.add(word.lower())
vocab = {word: i+1 for i, word in enumerate(sorted(list(words)))}
vocab['<PAD>'] = 0
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

# Define a fixed maximum length
max_len = 10

def sentence_to_indices(sentence):
    tokens = sentence.lower().split()
    # Truncate if too long, pad if too short
    tokens = tokens[:max_len]
    indices = [vocab[word] for word in tokens]
    indices += [vocab['<PAD>']] * (max_len - len(indices))
    return indices

# Convert sentences to tensor of indices
real_data = torch.tensor([sentence_to_indices(s) for s in sentences], dtype=torch.long)

# --- Hyperparameters ---
embedding_dim = 32
hidden_dim = 64
noise_dim = 16
batch_size = real_data.size(0)
device = torch.device("cpu")  # Use CPU for this setup

# --- Model Definitions ---
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, vocab_size, embedding_dim, max_len):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.fc = nn.Linear(noise_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, noise):
        h0 = torch.tanh(self.fc(noise)).unsqueeze(0)  # (1, batch, hidden_dim)
        c0 = torch.zeros_like(h0)
        # Start with a <PAD> token (acting as a start token)
        inputs = self.embed(torch.full((noise.size(0), 1), vocab['<PAD>'], dtype=torch.long, device=noise.device))
        outputs = []
        hidden = (h0, c0)
        for _ in range(self.max_len):
            out, hidden = self.lstm(inputs, hidden)  # out: (batch, 1, hidden_dim)
            logits = self.fc_out(out.squeeze(1))       # (batch, vocab_size)
            token_probs = F.gumbel_softmax(logits, tau=1, hard=True)
            outputs.append(token_probs.unsqueeze(1))
            token_indices = token_probs.argmax(dim=1)
            inputs = self.embed(token_indices).unsqueeze(1)
        outputs = torch.cat(outputs, dim=1)  # (batch, max_len, vocab_size)
        return outputs

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_len):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # If input is a probability distribution, convert it to token indices via argmax.
        if x.dim() == 3:
            x = x.argmax(dim=2)
        embedded = self.embed(x)
        _, (hn, _) = self.lstm(embedded)
        hn = hn.squeeze(0)
        out = self.fc(hn)
        return torch.sigmoid(out)

# --- Initialize Distributed Environment ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Master address (for multi-node, use actual IP)
    os.environ['MASTER_PORT'] = '12355'      # Port for communication
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# --- Training Loop with DistributedDataParallel ---
def train(rank, world_size):
    # Set up distributed environment
    setup(rank, world_size)
    
    # Initialize models
    generator = Generator(noise_dim, hidden_dim, vocab_size, embedding_dim, max_len).to(rank)
    discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim, max_len).to(rank)
    
    # Wrap models with DDP
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])
    
    # Optimizers and Loss
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.004)
    criterion = nn.BCELoss()

    # --- Data Preparation ---
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a distributed sampler for the dataset
    sampler = DistributedSampler(real_data, num_replicas=world_size, rank=rank)

    # DataLoader with distributed sampler
    data_loader = DataLoader(CustomDataset(real_data), batch_size=batch_size, sampler=sampler)
    
    # --- Training Loop ---
    num_epochs = 2000
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        for data in data_loader:
            data = data.to(rank)
            real_labels = torch.ones(data.size(0), 1, device=rank)
            fake_labels = torch.zeros(data.size(0), 1, device=rank)

            # --- Train Discriminator ---
            disc_optimizer.zero_grad()
            real_preds = discriminator(data)
            real_loss = criterion(real_preds, real_labels)
            
            noise = torch.randn(data.size(0), noise_dim, device=rank)
            fake_data = generator(noise)
            fake_preds = discriminator(fake_data.detach())
            fake_loss = criterion(fake_preds, fake_labels)
            
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            disc_optimizer.step()
            
            # --- Train Generator ---
            gen_optimizer.zero_grad()
            fake_preds = discriminator(fake_data)
            gen_loss = criterion(fake_preds, real_labels)  # Generator tries to fool the discriminator
            gen_loss.backward()
            gen_optimizer.step()
            
            if (epoch+1) % 200 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f}")

    # Clean up distributed environment
    cleanup()

# --- Run the Training on Multiple Nodes ---
if __name__ == '__main__':
    world_size = 2  # Number of processes (for example, 2 nodes)
    rank = int(os.environ['RANK'])
    train(rank, world_size)

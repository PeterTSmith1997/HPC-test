import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Instantiate models
generator = Generator(noise_dim, hidden_dim, vocab_size, embedding_dim, max_len).to(device)
discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim, max_len).to(device)

# --- Optimizers and Loss ---
gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Move real data to device and prepare labels
real_data = real_data.to(device)
real_labels = torch.ones(batch_size, 1, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)

# --- Training Loop ---
num_epochs = 2000
for epoch in range(num_epochs):
    # --- Train Discriminator ---
    disc_optimizer.zero_grad()
    real_preds = discriminator(real_data)
    real_loss = criterion(real_preds, real_labels)
    
    noise = torch.randn(batch_size, noise_dim, device=device)
    fake_data = generator(noise)
    fake_preds = discriminator(fake_data.detach())
    fake_loss = criterion(fake_preds, fake_labels)
    
    disc_loss = real_loss + fake_loss
    disc_loss.backward()
    disc_optimizer.step()
    
    # --- Train Generator ---
    gen_optimizer.zero_grad()
    noise = torch.randn(batch_size, noise_dim, device=device)
    fake_data = generator(noise)
    fake_preds = discriminator(fake_data)
    gen_loss = criterion(fake_preds, real_labels)  # Generator tries to fool the discriminator
    gen_loss.backward()
    gen_optimizer.step()
    
    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f}")

# --- Generate New Sentences ---
noise = torch.randn(5, noise_dim, device=device)
generated = generator(noise)
generated_indices = generated.argmax(dim=2).cpu().numpy()

# Build inverse vocabulary mapping
inv_vocab = {v: k for k, v in vocab.items()}
generated_sentences = []
for idx_seq in generated_indices:
    words_seq = [inv_vocab.get(idx, "") for idx in idx_seq]
    generated_sentences.append(" ".join(words_seq))
    
print("\nGenerated Sentences:")
for sent in generated_sentences:
    print(sent)

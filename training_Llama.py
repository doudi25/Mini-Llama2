import torch
from torch import nn
from torch.nn import functional as F
from handling_dataset import LlamaDataset
from mini_Llama import Transformer, ModelArgs
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

tokenizer = tiktoken.get_encoding('gpt2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = LlamaDataset(load_dataset("stanfordnlp/imdb")['train'] ,512)

data_loader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)

val_dataset = LlamaDataset(load_dataset("stanfordnlp/imdb")['test'],seq_len=512)

val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=2,shuffle=True)

def generation(max_tokens: int,model, p: float):

    start_token = torch.tensor(50258)

    x = start_token

    x = x.view(-1, 1).to(device)

    for _ in range(max_tokens):

        out = model(x)

        logits = out[:, -1, :]

        logits = F.softmax(logits, dim=-1)

        prob_sort, indices = torch.sort(logits, descending=True)

        cumulative_sum = torch.cumsum(prob_sort, dim=-1)

        mask = cumulative_sum <= p
        # Zero out probabilities outside the cumulative sum threshold
        masked_probs = torch.where(mask, prob_sort, torch.tensor(0.0, device=device))

        masked_probs_sum = masked_probs.sum(dim=-1, keepdim=True)

        # Normalize the masked probabilities without softmax
        if masked_probs_sum.item() > 0:

            normalized_probs = masked_probs / masked_probs_sum

            top_prob = torch.multinomial(normalized_probs, 1)

            index = torch.gather(indices, -1, top_prob)
        else:
            # If no probabilities exceed the threshold, fallback to a default token
            index = torch.tensor([50258], device=device)

            # Append the sampled token to the sequence
        x = torch.cat([x, index], dim=-1)
    return x

def decoding(tokens: torch.tensor):
    s = ''
    replic = []

    tokens = tokens.view(-1)

    for i in range(tokens.shape[0]):
        # tokenizer does not contain token of <sos>
        if tokens[i].item() == 50258:
            s += '<sos>'
        else:
            replic.append(tokens[i].item())

    gen_str = tokenizer.decode(replic)

    gen_str = s + gen_str
    return gen_str
# make the computation fast
torch.set_float32_matmul_precision('high')
model = Transformer(ModelArgs()).to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
tokens = generation(15, model, 0.6)
# if your cuda device supports torch compile apply it in the next line
# model = torch.compile(model)
print(f'Ai language generation before training : {decoding(tokens)}')
def training_loop(num_epochs):

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # make the loss ignore the padding index
    loss_fn = nn.CrossEntropyLoss(ignore_index=50257)
    losses = []
    for epoch in range(num_epochs):
        # make tqdm bar
        batch_iterator = tqdm(data_loader, desc=f'Processing epoch{epoch:02d}')
        for batch in batch_iterator:
            optimizer.zero_grad()
            # load input batch
            input = batch['decoder_input'].to(device)
            # load output batch
            output = batch['decoder_output'].to(device)
            # load padding mask for attention
            mask = batch['decoder_mask'].to(device)
            pred = model(input, mask)
            # convert pred from (B,seq_len,vocab_size) -> (B*seq_len,vocab_size)
            # convert output to (B*seq_len)
            loss = loss_fn(pred.view(pred.shape[0]*pred.shape[1], -1), output.view(-1))
            # run backpropagation
            loss.backward()
            losses.append(loss)
            # update the weights
            optimizer.step()
        if epoch % 4 == 0 or epoch == 0:
            # test the model
            val_loss = validation(val_loader, model)
            gen_token = generation(15, model, 0.6)
            gen_str = decoding(gen_token)
            print(f'Epoch {epoch} | Val-Loss {val_loss} | Generative-Text {gen_str}')


def validation(val_data, model):
    model.eval()  # Set model to evaluation mode
    batch_iterator = tqdm(val_data, desc='Processing validation')
    loss_test = nn.CrossEntropyLoss(ignore_index=50257)
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in batch_iterator:
            input = batch['decoder_input']
            output = batch['decoder_output']
            mask = batch['decoder_mask']

            # Move tensors to the same device as the model
            input = input.to(device)
            output = output.to(device)
            mask = mask.to(device)

            pred = model(input, mask)

            # Reshape predictions and targets
            pred = pred.view(-1, pred.shape[-1])
            output = output.view(-1)

            # Calculate loss
            loss_temp = loss_test(pred, output)
            total_loss += loss_temp.item()

            # Count the non-ignored tokens
            total_tokens += output.ne(50257).sum().item()

    # Average loss per token (excluding ignored tokens)
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss
training_loop(2)

torch.save(model.state_dict(),'Llama2 Mini')





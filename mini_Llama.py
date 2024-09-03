import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 16   # nbr query heads , it differs from k,v because we will use grouped query attention
    n_kv_heads: Optional[int] = None
    vocab_size: int = 50259   # unkown , will shosen in the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 20
    max_seq_len: int = 512
    device: str = None  # i do not specify cuda because if you want to run it on your own cpu you can
class selfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        # the number of query heads
        self.n_heads_q = args.n_heads

        self.d_k = args.dim // args.n_heads

        self.seq_len = args.max_seq_len


        self.Q_Proj = nn.Linear(args.dim,args.n_heads*self.d_k, bias=False)

        self.K_Proj = nn.Linear(args.dim,args.n_heads * self.d_k,bias=False)

        self.V_Proj = nn.Linear(args.dim,args.n_heads * self.d_k,bias=False)

        self.Out = nn.Linear(args.n_heads * self.d_k,args.dim,bias=False)


    def forward(self, x:torch.Tensor, freq_complex: torch.Tensor,mask=None):

        batch_size, seq_len, _ = x.shape # extract the shape of the input tensor which is batch by seq_len by embedding_dim

        query = self.Q_Proj(x) # convert the dim of x into B,seq_len,query_dim*n_heads

        key = self.K_Proj(x)  # the same dim of the first transformation but may differ in the last dim if we apply grouped query

        value = self.V_Proj(x) # same dim of key

        query = query.view(batch_size, seq_len, self.n_heads_q, self.d_k)

        key = key.view(batch_size, seq_len, self.n_heads_q, self.d_k)

        value = value.view(batch_size, seq_len, self.n_heads_q, self.d_k)

        query = apply_rotary_embeddings(query, freq_complex, device=query.device)

        key = apply_rotary_embeddings(key, freq_complex, device=key.device)

        query = query.transpose(1, 2)

        key = key.transpose(1, 2)

        value = value.transpose(1, 2)
        # apply flash attention

        with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
        ):
            # Calculate scaled dot-product attention
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)

            # Apply mask if provided
            if mask is not None:
                mask = mask.view(mask.shape[0], 1, 1, mask.shape[1])
                mask = (mask == 0)
                scores = scores.masked_fill(mask, 1e-9)

            # Softmax normalization
            scores = F.softmax(scores.float(), dim=-1).type_as(query)
            scores = F.softmax(scores.float(), dim=-1).type_as(query)

            output = torch.matmul(scores, value)

        output = (output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))

        return self.Out(output)
def precompute_theta_pos_frequencies(head_dim: int,seq_len: int,device: str, theta : float =1000.0):
    # embedding dim must me even
    assert head_dim % 2 == 0, "Dimension is even"
    theta_numerator = torch.arange(0,head_dim,2).float()

    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # build the m parameter
    m = torch.arange(seq_len, device=device)
    # the outer product is basically the multiplication of m.T with shape seq_len,1  with theta 1,m.shape[1]
    freq = torch.outer(m, theta).float()
    freq_complex = torch.polar(torch.ones_like(freq), freq)
    return freq_complex

def apply_rotary_embeddings(x:torch.Tensor,freq_complex:torch.Tensor,device:str):
    # convert it to complex number real + i*img
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # tackle dimensions
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    # element wise multiplication
    x_rotated = x_complex*freq_complex.to(device)
    # convert it to real and tackle the shape
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
# custom implementation of rms normalization
class RMSNormalization(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # intialize gamma (make it parameter to run backprop on it)
        self.gamma = nn.Parameter(torch.ones(dim))
    def normalization(self,x:torch.Tensor):
        # x / (rms(x) + eps)
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)

    def forward(self, x: torch.Tensor):
        # x * g

        return self.gamma * self.normalization(x.float()).type_as(x)
class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim

        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:

            hidden_dim = int(args.ffn_dim_multiplier*hidden_dim)

        hidden = args.multiple_of * ((hidden_dim+args.multiple_of-1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim,bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim,bias=False)
    def forward(self, x):
        # pass it to linear layer and apply silu activation
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        # element wise multiplication
        x = swish * x_v
        # reproject into embed_dim
        x = self.w2(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads

        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads

        self.attention = selfAttention(args)

        self.feed_forward = FeedForward(args)

        self.pre_norm = RMSNormalization(args.dim,eps=args.norm_eps)

        self.feedforward_norm = RMSNormalization(args.dim,eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freq_complex: torch.tensor, mask=None):

        hidden_states = x + self.attention(self.pre_norm(x), freq_complex, mask)
        out = hidden_states + self.feed_forward.forward(self.feedforward_norm(hidden_states))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, " Vocab must be set"

        self.args = args

        self.vocab_size = args.vocab_size

        self.n_layers = args.n_layers

        self.tokens_embeddings = nn.Embedding(self.vocab_size, args.dim)
        # build series of decoder block
        self.layers = nn.ModuleList([DecoderBlock(args) for _ in range(self.n_layers)])

        self.norm = RMSNormalization(args.dim, eps=args.norm_eps)
        # linear projection to embed_dim
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        # tie weight between token_embed and output  layers
        self.tokens_embeddings.weight = self.output.weight
        # compute freqs to use it in rotary embedding
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads,self.args.max_seq_len,device=self.args.device)

    def forward(self,tokens: torch.tensor,mask=None):
        # (B,Seq_len) -> #(B,Seq_len,embed_dim)
        batch_size, seq_len = tokens.shape

        if seq_len < self.args.max_seq_len:
            # tackle the freqs dim problem if we have generation mode not training with fixed seq_len

            self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, seq_len, device=device)
        # embed input tokens
        h = self.tokens_embeddings(tokens)
        freq_complex = self.freqs_complex
        for layer in self.layers:
            h = layer(h, freq_complex, mask)
            h = self.norm(h)
        # last decoder head we project the output to vocab_size
        out = self.output(h).float()
        return out


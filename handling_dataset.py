from torch.utils.data import Dataset
import torch
import torch
from torch.utils.data import Dataset
import tiktoken
from torch.nn import functional as F
class LlamaDataset(Dataset):
    def __init__(self, dataset, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.data = list(self.get_all_sentences(dataset))

    def get_all_sentences(self, dataset):
        for item in dataset:
            yield item['text']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = self.data[index]
        #input_tokens = self.tokenizer.encode(input_text).ids
        input_tokens = self.tokenizer.encode(input_text)

        # Ensure sequence length
        if len(input_tokens) >= self.seq_len - 2:
            input_tokens = input_tokens[:self.seq_len - 2]
        # calculate the padding (seq_len - input_length - 2 )  the 2 is corresponding to start of sentence and end of sentence tokens
        padding_tokens = self.seq_len - len(input_tokens) - 2

        if padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Convert to tensors and pad appropriately
        sos_ids = torch.tensor([50258])
        pad_ids = torch.tensor([50257])
        eos_ids = self.tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})
        eos_ids = torch.tensor(eos_ids)

        input_tensor = torch.cat([
            sos_ids.unsqueeze(0),  # Add batch dimension
            torch.tensor(input_tokens).unsqueeze(0),
            pad_ids.unsqueeze(0).repeat(1, padding_tokens)  # Repeat pad_ids for padding
        ], dim=1).squeeze(0)  # Remove batch dimension for final tensor

        target_tensor = torch.cat([
            input_tensor[1:],
            eos_ids
        ])
        # Create attention mask

        attention_mask = (input_tensor != 50257).int()
        return {
            "decoder_input": input_tensor,
            "decoder_output": target_tensor,
            "decoder_mask": attention_mask
        }


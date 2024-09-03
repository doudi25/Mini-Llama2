# Mini-Llama2
-Custom impelementation of llama2 without kv cache , i used imdb dataset for training
if you want to train the model you can play with paramters of optimizer and embedding dim , and training epochs without touching tokenizer and dataloader .
-I used top p sampling in the generation , and the gp2 tokenizer from tiktoken library (openai) . i do not suggest to train the model on cpu device .


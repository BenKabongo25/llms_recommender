# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Attn2Seq: https://aclanthology.org/E17-1059/


import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *


class AttributeEncoder(nn.Module):

    def __init__(self, n_users: int, n_items: int, args: Any):
        super().__init__()
        self.args = args
        self.n_users = n_users
        self.n_items = n_items
        self.n_attributes = 0

        if self.args.user_id_flag:
            self.n_attributes += 1
            self.user_embed = nn.Embedding(
                num_embeddings=self.n_users + 1, 
                embedding_dim=self.args.attribute_embedding_dim,
                padding_idx=self.args.padding_idx
            )

        if self.args.item_id_flag:
            self.n_attributes += 1
            self.item_embed = nn.Embedding(
                num_embeddings=self.n_items + 1, 
                embedding_dim=self.args.attribute_embedding_dim,
                padding_idx=self.args.padding_idx
            )

        if self.args.rating_flag:
            self.n_attributes += 1
            self.rating_embed = nn.Sequential(
                nn.Linear(1, self.args.attribute_embedding_dim)
            )

        assert self.n_attributes > 0, "Number of attributes must be > 0"

        self.fc = nn.Sequential(
            nn.Linear(
                self.n_attributes * self.args.attribute_embedding_dim,
                self.args.n_layers * self.args.hidden_dim
            ),
            nn.Tanh()
        )

    def forward(
        self, 
        U_ids: torch.Tensor=None, I_ids: torch.Tensor=None, ratings: torch.Tensor=None
    ) -> torch.Tensor:
        embeddings_list = []

        if self.args.user_id_flag:
            assert U_ids is not None, "You must pass users ids"
            U_embeddings = self.user_embed(U_ids)
            embeddings.append(U_embeddings)

        if self.args.item_id_flag:
            assert I_ids is not None, "You must pass items ids"
            I_embeddings = self.item_embed(I_ids)
            embeddings.append(I_embeddings)

        if self.args.rating_flag:
            assert ratings is not None, "You must pass ratings"
            R_embeddings = self.rating_embed(ratings)
            embeddings.append(R_embeddings)

        embeddings = torch.cat(embeddings_list, dim=-1)
        out = self.fc(embeddings)
        return out, embeddings_list


class SequenceDecoder(nn.Module):

    def __init__(self, vocab_size: int, args: Any):
        super().__init__()   
        self.args = args

        self.vocab_size = vocab_size
        self.vocab_embed = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.args.vocab_embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.rnns = nn.LSTM(
            input_size=self.args.hidden_dim,
            hidden_size=self.args.hidden_dim,
            num_layers=self.args.n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.args.hidden_dim, self.vocab_size)
        
    def forward(
        self, 
        X: torch.Tensor, h_0: torch.Tensor=None, c_0: torch.Tensor=None
    ) -> torch.Tensor:
        args = {}
        if h_0 is not None:
            args["h_0"] = h_0
        else:
            args["c_0"] = c_0

        embeddings = self.vocab_embed(X)
        _, h_n = self.rnns(input=embeddings, **args)
        output = self.fc(h_n[-1])
        return h_n, output

    def decode(
        self, 
        h_t: torch.Tensor, target: torch.Tensor=None, 
        teach_forcing_flag: bool=False, 
        train_flag: bool=False
    ) -> torch.Tensor:
        index = 0
        batch_size = h_t.size(1)
        eos_cpt = 0
        logits = []

        input = torch.LongTensor([self.args.sos_token_id]).repeat(batch_size).to(h_t.device)
        while index != self.args.max_length:
            hidden, output = self.forward(input, hidden)
            logits.append(output)
            if not teach_forcing_flag:
                input = torch.softmax(output, dim=-1).argmax(dim=-1)[0]
                if input.size(0) > 1: input = input.squeeze()
            else:
                input = target[:, index]
            eos_cpt += torch.sum(input == self.eos_idx).item()
            if not train_flag and eos_cpt == batch_size:
                break
            index += 1

        return torch.cat(logits, dim=0).transpose(0, 1)


class Attn2Seq(nn.Module):

    def __init__(self, n_users: int, n_items: int, vocab_size: int, args: Any):
        super().__init__()
        self.attribute_encoder = AttributeEncoder(n_users, n_items, args)
        self.sequence_decoder = SequenceDecoder(vocab_size, args)

    def forward()

def main(args):
    # TODO
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--user_id_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(user_id_flag=True)
    parser.add_argument("--item_id_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(item_id_flag=True)
    parser.add_argument("--rating_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(rating_flag=False)
    parser.add_argument("--attention_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(attention_flag=True)

    parser.add_argument("--attribute_embedding_dim", type=int, default=32)
    parser.add_argument("--vocab_embedding_dim", type=int, default=32)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=4)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--sos_token_id", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=2)
    parser.add_argument("--oov_token_id", type=int, default=3)



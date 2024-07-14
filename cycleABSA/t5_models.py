# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Models

import torch.optim as optim
import lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Any, Dict, List, Tuple, Union


class T5FineTuner(pl.LightningModule):

    def __init__(self, tokenizer: TimeoutError, args: Any):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name_or_path)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def _step(self, batch):
        input_ids, attention_mask, labels, annotations = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        return loss

    def configure_optimizers(self):
        optimzier = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimzier
    

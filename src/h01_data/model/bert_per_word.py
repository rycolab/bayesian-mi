import torch
import torch.nn as nn
from transformers import BertModel

from util import constants


class BertPerWordModel(nn.Module):
    # pylint: disable=arguments-differ

    def __init__(self, bert_option):
        super().__init__()
        self.bert = self.get_bert(bert_option)

    @staticmethod
    def get_bert(bert_option):
        model = BertModel.from_pretrained(bert_option)
        return model

    def forward(self, x, attention_mask, mappings):
        outputs = self.bert(x, attention_mask=attention_mask)
        last_layer = outputs[0]
        return self.from_bpe_to_word(last_layer, mappings)

    def from_bpe_to_word(self, output, mappings):
        batch_size = output.size(0)
        longest_token_sent = mappings.size(1)

        hidden_states = output[:, 1:-1]
        embedding_size = output.size(-1)

        hidden_states_per_token = torch.zeros(
            (batch_size, longest_token_sent, embedding_size)).to(device=constants.device)
        mask_start = torch.zeros(batch_size).long().to(device=constants.device)

        for mask_pos in range(0, longest_token_sent):
            mask_sizes = mappings[:, mask_pos]

            hidden_states_per_token[:, mask_pos] = \
                self.sum_bpe_embeddings(hidden_states, mask_start, mask_sizes)

            mask_start += mask_sizes

        return hidden_states_per_token

    @staticmethod
    def sum_bpe_embeddings(hidden_states, mask_start, mask_sizes):
        mask_idxs = []
        for i, (sent_start, sent_size) in enumerate(zip(mask_start, mask_sizes)):
            mask_idxs += [(i, sent_start.item() + x) for x in range(sent_size)]
        mask_idxs = list(zip(*mask_idxs))

        hidden_states_temp = \
            torch.zeros_like(hidden_states).float().to(device=constants.device)
        hidden_states_temp[mask_idxs] = hidden_states[mask_idxs]

        embedding_size = hidden_states.size(-1)
        return hidden_states_temp.sum(dim=1) / \
            mask_sizes.unsqueeze(-1).repeat(1, embedding_size).float()

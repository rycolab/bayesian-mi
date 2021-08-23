import numpy as np
import torch
from transformers import BertTokenizer

from h01_data.model import BertPerWordModel
from util import constants
from util import util
from .ud import UdProcessor


class BertProcessor(UdProcessor):
    # pylint: disable=arguments-differ
    bert_name = 'bert-base-multilingual-cased'
    name = 'bert'

    def __init__(self):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self.bert_model = BertPerWordModel(self.bert_name).to(device=constants.device)
        self.bert_model.eval()

        self.pad_id = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')

    def process_file(self, ud_file, output_file, batch_size=8):
        print("Processing file {}".format(ud_file))

        print("PHASE ONE: reading file and tokenizing")
        all_bert_tokens, all_bert2target_map = self.tokenize(ud_file)

        print("PHASE TWO: padding, batching, and embedding for bert")
        all_bert_embeddings = self.embed_bert(all_bert_tokens, all_bert2target_map, batch_size)

        util.write_data(output_file % self.name, all_bert_embeddings)

        print("Completed {}".format(ud_file))

    def tokenize(self, file_name):
        all_ud_tokens, _ = super().tokenize(file_name)

        all_bert_tokens = []
        all_bert2target_map = []

        # Initialise all the trees and embeddings
        for ud_tokens in all_ud_tokens:

            # Tokenize the sentence
            ud2bert_mapping = []
            bert_tokens = []
            for token in self.iterate_sentence(ud_tokens):
                bert_decomposition = self.bert_tokenizer.tokenize(token)
                if len(bert_decomposition) == 0:
                    bert_decomposition = ['[UNK]']

                bert_tokens += bert_decomposition
                ud2bert_mapping.append(len(bert_decomposition))

            all_bert2target_map.append(ud2bert_mapping)
            all_bert_tokens.append(bert_tokens)

        return all_bert_tokens, all_bert2target_map

    @staticmethod
    def iterate_sentence(tokens):
        return tokens

    def embed_bert(self, all_bert_tokens, all_mappings, batch_size):
        all_bert_embeddings = []

        batch_num = 0
        for batch_start in range(0, len(all_bert_tokens), batch_size):

            batch_num += 1
            if batch_num % 10 == 0:
                print("Processing batch {} to embeddings".format(batch_num))

            # Get the batch
            batch_end = batch_start + batch_size
            batch = all_bert_tokens[batch_start:batch_end]
            batch_map = all_mappings[batch_start:batch_end]

            all_bert_embeddings += self.embed_batch(batch, batch_map)

        return all_bert_embeddings

    def embed_batch(self, batch, batch_map):
        input_ids, attention_mask, mappings, lengths = \
            self.get_batch_tensors(batch, batch_map)

        with torch.no_grad():
            embeddings = self.bert_model(input_ids, attention_mask, mappings)

        last_hidden_states = [
            x[:lengths[i]]
            for i, x in enumerate(embeddings.cpu().numpy())
        ]

        return last_hidden_states

    def get_batch_tensors(self, batch, batch_map):
        lengths_bert = [(len(sentence) + 2) for sentence in batch]  # +2 for CLS/SEP
        longest_sent_bert = max(lengths_bert)
        lengths_orig = [(len(sentence)) for sentence in batch_map]
        longest_sent_orig = max(lengths_orig)

        # Pad it & build up attention mask
        input_ids = np.ones((len(batch), longest_sent_bert)) * self.pad_id
        attention_mask = np.zeros((len(batch), longest_sent_bert))
        mappings = np.ones((len(batch), longest_sent_orig)) * -1

        for i, sentence in enumerate(batch):
            sentence_len = lengths_bert[i]

            input_ids[i, :sentence_len] = self.get_sentence_ids(sentence)
            # Mask is 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            attention_mask[i, :sentence_len] = 1
            mappings[i, :len(batch_map[i])] = batch_map[i]

        # Move data to torch and cuda
        input_ids = torch.LongTensor(input_ids).to(device=constants.device)
        attention_mask = torch.LongTensor(attention_mask).to(device=constants.device)
        mappings = torch.LongTensor(mappings).to(device=constants.device)

        return input_ids, attention_mask, mappings, lengths_orig

    def get_sentence_ids(self, sentence):
        return self.bert_tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + sentence + ["[SEP]"])

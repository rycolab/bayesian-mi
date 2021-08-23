from transformers import AlbertTokenizer

from h01_data.model import AlbertPerWordModel
from util import constants
from .bert import BertProcessor


class AlbertProcessor(BertProcessor):
    # pylint: disable=arguments-differ
    albert_name = 'albert-base-v2'
    name = 'albert'

    def __init__(self):
        super().__init__()
        self.bert_tokenizer = AlbertTokenizer.from_pretrained(self.albert_name)
        self.bert_model = AlbertPerWordModel(self.albert_name).to(device=constants.device)
        self.bert_model.eval()

        self.pad_id = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')

from transformers import RobertaTokenizer

from h01_data.model import RobertaPerWordModel
from util import constants
from .bert import BertProcessor


class RobertaProcessor(BertProcessor):
    # pylint: disable=arguments-differ
    roberta_name = 'roberta-base'
    name = 'roberta'

    def __init__(self):
        super().__init__()
        print('roberta')
        self.bert_tokenizer = RobertaTokenizer.from_pretrained(self.roberta_name)
        self.bert_model = RobertaPerWordModel(self.roberta_name).to(device=constants.device)
        self.bert_model.eval()

        self.pad_id = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')

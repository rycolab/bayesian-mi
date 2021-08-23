from transformers import RobertaModel

from .bert_per_word import BertPerWordModel


class RobertaPerWordModel(BertPerWordModel):
    @staticmethod
    def get_bert(bert_option):
        model = RobertaModel.from_pretrained(bert_option)
        return model

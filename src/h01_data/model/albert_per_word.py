from transformers import AlbertModel

from .bert_per_word import BertPerWordModel


class AlbertPerWordModel(BertPerWordModel):
    @staticmethod
    def get_bert(bert_option):
        model = AlbertModel.from_pretrained(bert_option)
        return model

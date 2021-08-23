import os
import fasttext
import fasttext.util

from util import constants
from util import util
from .ud import UdProcessor


class FasttextProcessor(UdProcessor):
    def __init__(self, language):
        super().__init__()
        self.fasttext_model = self.load_model(language)

    @staticmethod
    def load_model(language):
        lang = constants.LANGUAGE_CODES[language]
        ft_path = 'data/fasttext'
        ft_fname = os.path.join(ft_path, 'cc.%s.300.bin' % lang)
        if not os.path.exists(ft_fname):
            print("Downloading fasttext model")
            temp_fname = fasttext.util.download_model(lang, if_exists='ignore')
            util.mkdir(ft_path)
            os.rename(temp_fname, ft_fname)
            os.rename(temp_fname + '.gz', ft_fname + '.gz')

        print("Loading fasttext model")
        return fasttext.load_model(ft_fname)

    def process_file(self, ud_file, output_file, **kwargs):
        print("Processing file {}".format(ud_file))

        print("PHASE ONE: reading file and tokenizing")
        tokens, _ = self.tokenize(ud_file)

        print("PHASE FOUR: getting fasttext embeddings")
        fast_embeddings = self.get_embeddings(tokens)

        util.write_data(output_file % 'fast', fast_embeddings)

        print("Completed {}".format(ud_file))

    def get_embeddings(self, words):
        embeddings = [[self.fasttext_model[word] for word in sentence]for sentence in words]
        return embeddings

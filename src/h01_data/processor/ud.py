from conllu import parse_incr

from util import util


class UdProcessor:
    def __init__(self):
        self.max_tokens = 100

    def process_file(self, ud_file, output_file, **kwargs):
        # pylint: disable=unused-argument
        print("Processing file {}".format(ud_file))

        print("PHASE ONE: reading file and tokenizing")
        tokens, ud_data = self.tokenize(ud_file)
        save_data = list(zip(ud_data, tokens))

        # Pickle, compress, and save
        util.write_data(output_file % 'ud', save_data)

    def tokenize(self, file_name):
        all_ud_tokens = []
        all_ud_data = []

        count_del, count_total = 0, 0

        # Initialise all the trees and embeddings
        with open(file_name, "r", encoding="utf-8") as file:
            for token_list in parse_incr(file):

                ud_tokens = []
                ud_data = []

                for item in token_list:
                    ud_tokens.append(item['form'])
                    ud_data.append({
                        'word': item['form'],
                        'pos': item['upostag'],
                        'head': item['head'],
                        'rel': item['deprel'],
                    })

                # If there are more than max_tokens tokens skip the sentence
                if len(ud_tokens) <= self.max_tokens:
                    all_ud_tokens.append(ud_tokens)
                    all_ud_data.append(ud_data)
                else:
                    count_del += 1
                count_total += 1

        if count_del > 0:
            print('\n\n\tWarning!Removed %d (of %d) long sentences\n\n' % (count_del, count_total))
        return all_ud_tokens, all_ud_data

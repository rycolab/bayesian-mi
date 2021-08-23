import os
import sys
import argparse
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.processor import \
    UdProcessor, FasttextProcessor, \
    BertProcessor, AlbertProcessor, RobertaProcessor
from util import util
from util.ud_list import UD_LIST


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--batch-size",
                        help="The size of the mini batches",
                        default=8,
                        required=False,
                        type=int)
    parser.add_argument("--language",
                        help="The language to use",
                        required=True,
                        type=str)
    parser.add_argument("--representation", type=str, required=True)
    parser.add_argument("--ud-path",
                        help="The path to raw ud data",
                        default='data/ud/ud-treebanks-v2.5/',
                        required=False,
                        type=str)
    parser.add_argument("--output-path",
                        help="The path to save processed data",
                        default='data/processed/',
                        required=False,
                        type=str)
    args = parser.parse_args()
    print(args)

    return args


def get_ud_file_base(ud_path, language):
    return os.path.join(ud_path, UD_LIST[language])


def get_data_file_base(output_path, language):
    output_path = os.path.join(output_path, language)
    util.mkdir(output_path)
    return os.path.join(output_path, '%s--%s.pickle.bz2')


def get_data_processor(representation, args):
    if representation in ['ud', 'random', 'onehot']:
        processor = UdProcessor()
    elif representation == 'bert':
        processor = BertProcessor()
    elif representation == 'albert':
        processor = AlbertProcessor()
    elif representation == 'roberta':
        processor = RobertaProcessor()
    elif representation == 'fast':
        processor = FasttextProcessor(args.language)
    else:
        raise ValueError('Invalid representation %s' % representation)

    return processor


def process(language, ud_path, batch_size, output_path, representation, args):
    print("Loading data processor")
    processor = get_data_processor(representation, args)

    print("Precessing language %s" % language)
    ud_file_base = get_ud_file_base(ud_path, language)
    output_file_base = get_data_file_base(output_path, language)

    for mode in ['train', 'dev', 'test']:
        ud_file = ud_file_base % mode
        output_file = output_file_base % (mode, '%s')
        processor.process_file(ud_file, output_file, batch_size=batch_size)

    print("Process finished")


def main():
    args = get_args()

    batch_size = args.batch_size
    language = args.language
    ud_path = args.ud_path
    output_path = args.output_path
    # args.bert_name =

    with torch.no_grad():
        process(language, ud_path, batch_size, output_path, args.representation, args)


if __name__ == "__main__":
    main()

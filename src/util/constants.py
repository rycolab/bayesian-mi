import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


USED_LANGUAGES = ['english', 'basque', 'finnish', 'marathi', 'turkish']

LANGUAGE_CODES = {
    'english': 'en',
    'czech': 'cs',
    'basque': 'eu',
    'finnish': 'fi',
    'turkish': 'tr',
    'arabic': 'ar',
    'japanese': 'ja',
    'tamil': 'ta',
    'korean': 'ko',
    'marathi': 'mr',
    'urdu': 'ur',
    'telugu': 'te',
    'indonesian': 'id',
}

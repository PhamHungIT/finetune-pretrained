import re

import visen
from unicodedata import normalize
from pyvi import ViTokenizer

### brackets list
opening_ls = ['[', '{', '⁅', '〈', '⎡', '⎢', '⎣', '⎧', '⎨', '⎩', '❬', '❰', '❲', '❴', '⟦', '⟨', '⟪', '⟬', '⦃', '⦇', '⦉',
              '⦋', '⦍', '⦏', '⦑', '⦓', '⦕', '⦗', '⧼', '⸂', '⸄', '⸉', '⸌', '⸜', '⸢', '⸤', '⸦', '〈', '《', '「', '『',
              '【', '〔', '〖', '〘', '〚', '﹛', '﹝', '［', '｛', '｢', '｣']

closing_ls = [']', '}', '⁆', '〉', '⎤', '⎥', '⎦', '⎫', '⎬', '⎭', '❭', '❱', '❳', '❵', '⟧', '⟩', '⟫', '⟭', '⦄', '⦈', '⦊',
              '⦌', '⦎', '⦐', '⦒', '⦔', '⦖', '⦘', '⧽', '⸃', '⸅', '⸊', '⸍', '⸝', '⸣', '⸥', '⸧', '〉', '》', '」', '』',
              '】', '〕', '〗', '〙', '〛', '﹜', '﹞', '］', '｝', '｣']

opening_brackets = {key: '(' for key in opening_ls}
closing_brackets = {key: ')' for key in closing_ls}

### constant
PUNC = '!\"#$&()*+,-–−./:;=?@[\]^_`{|}~”“`°²ˈ‐ㄧ‛∼’'  # remove <> for number_sym and unknown_sym
UNKNOWN_SYM = ''  # mix number with character, ex: 6.23A

re_email = '([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z]{2,4}(\\.?[a-zA-Z]{2,4})?)'
re_url = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
re_url2 = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
re_image = '(tập[\s_]tin|hình|file|image|imagesize).*?(jpg|svg|png|gif|jpeg|ogg|tif|width)'
re_num_and_decimal = '[0-9]*[,.\-]*[0-9]*[,.\-]*[0-9]*[.,\-]*[0-9]*[,.\-]*[0-9]+[.,]?'
re_unknown = '[a-z]+[\d]+[\w]*|[\d]+[a-z]+[\w]*'
re_vnese_txt = r'[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s]'
re_truncate_unknown = '(<UNKNUM>\s*)+'


special_punc = {'”': '"', '': '', "’": "'", "`": "'"}


def replace_all(replacer: dict, txt: str) -> str:
    for old, new in replacer.items():
        txt = txt.replace(old, new)
    return txt


def replace_num(text: str) -> str:
    text = re.sub(re_num_and_decimal, UNKNOWN_SYM, text)
    return text


def replace_unknown(text: str) -> str:
    text = re.sub(re_unknown, UNKNOWN_SYM, text)
    return text


def unicode_normalizer(text, forms: list = ['NFKC', 'NKFD', 'NFC', 'NFD']) -> str:
    for form in forms:
        text = normalize(form, text)
    return text


def normalize_bracket(text: str) -> str:
    text = replace_all(opening_brackets, text)
    text = replace_all(closing_brackets, text)
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text)
    return text


def remove_punc(text: str) -> str:
    r = re.compile(r'[\s{}]+'.format(re.escape(PUNC)))
    text = r.split(text)
    return ' '.join(i for i in text if i)


def truncate_unknown(text: str) -> str:
    text = re.sub(re_truncate_unknown, UNKNOWN_SYM, text)
    return text


with open('../data/stop_words.txt', 'r') as fi:
    stop_words = fi.readlines() 
    stop_words = [word.rstrip() for word in stop_words]


def clean_text(text: str, stop_words=stop_words) -> str:
    text = str(text)
    text = text.split('\n')[0]
    text = unicode_normalizer(text, ["NFKC"])
    text = text.lower()
    text = remove_punc(text)
    text = truncate_unknown(text)
    text = re.sub(re_vnese_txt, " ", text)
    text = text.strip()
    text = ViTokenizer.tokenize(text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return visen.clean_tone(text)

import re


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def is_number(s):
    number_regex = re.compile(
        r'^([\[\]\-()|/!º°¤.,;:$×+_?\d/\s]|a|c|e|h|l|o|q|x|kg|m?g|ml|ou|anos?|cep|dias?|pdf|\x01)+$', re.IGNORECASE)

    return bool(number_regex.match(s))


def is_link(s):
    link_regex = re.compile(r'^https?\S*')

    return bool(link_regex.match(s))


def remove_break_lines(text):
    unwanted_regex = re.compile(r'\n')

    return unwanted_regex.sub(' ', text).strip()

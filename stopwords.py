from hazm import stopwords_list


def get_stopwords():
    with open("stopwords.txt", 'r') as f:
        results = f.readline().split()
    return results


stopwords = stopwords_list()
punctuation = get_stopwords()

all_stopwords = punctuation + stopwords + ["NUM"] + ['آقا', 'آور', 'افزا', 'باش', 'بردار', 'بست', 'بند', 'توان',
                                                     'توانست', 'دارا', 'دان', 'ده', 'رس', 'ریخت', 'ریز', 'سال', 'سو',
                                                     'شخص', 'شو', 'هست', 'وقت', 'کس', 'کن', 'گذار', 'گذاشت', 'گرد',
                                                     'گشت', 'گو', 'گیر', 'یاب'] + ['بس']

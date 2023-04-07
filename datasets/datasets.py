import datasets

def get_iwslt_dataset(source_lang='en', target_lang='vi', year='2015'):
    return datasets.load_dataset("ted_talks_iwslt", language_pair=(source_lang, target_lang), year=year)
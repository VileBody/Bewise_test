import torch
import numpy as np
import pandas as pd
from yargy.tokenizer import MorphTokenizer
from yargy import Parser, rule, and_, not_, or_
from yargy.predicates import gram, is_capitalized, dictionary, eq, caseless
from yargy.pipelines import morph_pipeline, caseless_pipeline




class ParserConvs:
  def __init__(self):
    ENH_MODEL_DIR = 'snakers4/silero-models'
    ENH_MODEL_NAME = 'silero_te'
    NAMES_DATASET = 'https://github.com/VileBody/Bewise_test/raw/main/russian_names.csv'
    model, example_texts, languages, punct, self.apply_te = torch.hub.load(repo_or_dir=ENH_MODEL_DIR,
                                                                model=ENH_MODEL_NAME)
    self.rusnames = pd.read_csv(NAMES_DATASET, index_col=0)
    self.names = self.rusnames.loc[self.rusnames['PeoplesCount']>=150,'Name'].values
    self.letters = 'йцукенгшщзхъфывапролджэёячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮ%'

    NAME = rule(
        not_(gram('INTJ')), gram('Name')#фикс
    )
    self.pars_name = Parser(NAME)
    #здесь я полагаю, что человеческие имена с прописной буквы сетка, обученная на сколько ни будь больших данных могла научиться хорошо

    goodbye = ['До встречи', 'До свидания', 'Всего хорошего', 'Всего доброго' 'Всего наилучшего', 'Хорошего дня', 'Хорошего вечера']
    GOODBYE = rule(caseless_pipeline(goodbye))
    allg = rule('всего', 'доброго')
    finl = or_(GOODBYE, allg)
    self.gb_parser = Parser(finl)

    GREETING = or_(rule(
        caseless('Добрый'), gram('NOUN').optional() #добрый хочется сделать normalized но доброго вам дня на мой взгляд несколько устарело + возникают накладки со словами прощания
    ), rule(caseless('Здравствуйте')))

    self.gr_parser = Parser(GREETING)
    adj_noun = or_(rule(gram('NOUN'), gram('ADJF'), gram('NOUN')), 
    rule(gram('ADJF'), gram('NOUN')), 
    rule(gram('NOUN').repeatable(max=3)))
    cname_2 = rule(gram('Name'), eq(','), adj_noun)
    cname_1 = rule(caseless('компания'), adj_noun)
    self.cparser = Parser(or_(cname_1, cname_2))  
    self.nparser = Parser(rule(gram('Name')))

  def get_company_name(self, text):
    x = list(self.cparser.findall(text))
    if x:
      x = x[0]
      res = [y.value for y in x.tokens]
      res = ' '.join(res)
      if res:
        name = list(self.nparser.findall(res))
        if name:
          namesp = list([x.tokens[0].span for x in name if x.tokens[0].value in self.names][0])
          res = res[:namesp[0]] + res[namesp[1]:]
      res = ' '.join([x for x in res.split() if x!=','])
      return res 
    return 'No company name'

  def extract_greeting(self, text):
    x = list(self.gr_parser.findall(text))
    if x:
      res = ' '.join([' '.join([y.value for y in z.tokens]) for z in x])
      return res
    return 'No hello'

  def extract_goodbye(self, text):
    x = list(self.gb_parser.findall(text))
    if x:
      res = ' '.join([' '.join([y.value for y in z.tokens]) for z in x])
      return res
    return 'No goodbye'

  def extract_name(self, text):
      for occurrency in self.pars_name.findall(text):
        for token in occurrency.tokens:
          if token.value in self.names:
            return token.value
      return 'No name'   

  def enhance(self, input_text):
    # по какой то причине если добавить прописные буквы в текст все ломается
    #думаю нет никакой проблемы в том чтобы их убрать, поскольку они обозначают
    #лишь начало реплики, они обязательно восстановятся
    input_text = input_text.lower()
    
    if (len(input_text) >= 1) and (input_text[0] not in self.letters):
      if (len(input_text) >= 2) and (input_text[1] not in self.letters):
        if (len(input_text)>=3) and (input_text[2] not in self.letters):
          start = 0
        else:
          start = 2
      else:
        start = 1
    else:
      start = 0

    result = self.apply_te(input_text[start:], lan='ru')
    if start > 0:
      if len(result) == 1:
        result = result.lower()
      elif len(result) > 1:
        result = result[0].lower() + result[1:]

    result = input_text[:start] + result
    return result


  def parse(self, data):
    mng_data = data[data['role']=='manager'].drop('role', axis=1).reset_index(drop=True)
    mng_data['enhanced_text'] = mng_data['text'].apply(self.enhance)
    mng_data['name'] = mng_data['enhanced_text'].apply(lambda x: self.extract_name(x))
    mng_data['goodbye'] = mng_data['enhanced_text'].apply(lambda x: self.extract_goodbye(x))
    mng_data['greeting'] = mng_data['enhanced_text'].apply(self.extract_greeting)
    mng_data['company_name'] = mng_data['enhanced_text'].apply(self.get_company_name)  
    result = dict()
    result['greeting'] = mng_data.loc[mng_data['greeting']!='No hello', 'text'].tolist()
    result['named_herself'] = mng_data.loc[mng_data['name']!='No name', 'text'].tolist()[:1]
    result['name'] = mng_data.loc[mng_data['name']!='No name', 'name'].tolist()[:1]
    result['company_name'] =\
       mng_data.loc[mng_data['company_name']!='No company name', 'company_name'].tolist()
    result['goodbye'] = mng_data.loc[mng_data['goodbye']!='No goodbye', 'goodbye'].tolist()
    result['check_gr_db'] = (len(result['greeting']) > 0) and (len(result['goodbye']) > 0)
    return result

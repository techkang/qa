from stanfordcorenlp import StanfordCoreNLP
import googlesearch

nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05',lang='zh')

sentence='中科大成立于1958年9月20日早上8点12分36秒'

print(nlp.ner(sentence))
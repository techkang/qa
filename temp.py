# from stanfordcorenlp import StanfordCoreNLP
# import googlesearch

# nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05',lang='zh')

# sentence='中科大成立于1958年9月20日早上8点12分36秒'

# print(nlp.ner(sentence))

questions=[]
with open('questions.txt',encoding='utf-8') as f:
    question={}
    for line in f:
        line=line.strip().split()
        question["question"]=line[0].strip().split('.')[1]
        question['answers']=[line[1].strip()]
        questions.append(str(question))
print(',\n'.join(questions))
    
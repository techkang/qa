# from stanfordcorenlp import StanfordCoreNLP
# import googlesearch

# nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05',lang='zh')

# sentence='中科大成立于1958年9月20日早上8点12分36秒'

# print(nlp.ner(sentence))

questions = []
with open('questions.txt', encoding='utf-8') as f:
    
    for line in f:
        line = line.strip().split()
        question = {}
        question["question"] = line[0].strip().split('.')[1]
        question['answers'] = [line[1].strip()]
        questions.append(question)
for d in questions:
    print('\item \\textbf{{Q: }}{}\\textbf{{A: }}{}'.format(d['question'],d['answers'][0]))

'''
[
  {
    "question": "谁是中科大校长",
    "answers": [
      "包信和"
    ]
  },
  {
    "question": "包信和是哪里人",
    "answers": [
      "江苏扬中"
    ]
  },
  {
    "question": "包信和什么时候出生",
    "answers": [
      "1959年8月"
    ]
  },
  {
    "question": "潘建伟是哪里人",
    "answers": [
      "浙江东阳"
    ]
  },
  {
    "question": "潘建伟什么时候出生",
    "answers": [
      "1970年3月"
    ]
  },
  {
    "question": "谁是中科大校党委书记",
    "answers": [
      "舒歌群"
    ]
  },
  {
    "question": "舒歌群是哪里人",
    "answers": [
      "浙江湖州"
    ]
  },
  {
    "question": "舒歌群什么时候出生",
    "answers": [
      "1964年3月"
    ]
  },
  {
    "question": "中科大是什么时候建校的",
    "answers": [
      "1958年9月"
    ]
  },
  {
    "question": "朱清时是哪里人",
    "answers": [
      "四川成都"
    ]
  },
  {
    "question": "朱清时什么时候出生",
    "answers": [
      "1946年2月"
    ]
  },
  {
    "question": "郭沫若是哪里人",
    "answers": [
      "四川乐山"
    ]
  },
  {
    "question": "郭沫若什么时候出生",
    "answers": [
      "1892年11月"
    ]
  },
  {
    "question": "严济慈是哪里人",
    "answers": [
      "浙江东阳"
    ]
  },
  {
    "question": "严济慈什么时候出生",
    "answers": [
      "1901年1月"
    ]
  },
  {
    "question": "谁是科大第一任校长",
    "answers": [
      "郭沫若"
    ]
  },
  {
    "question": "侯建国是哪里人",
    "answers": [
      "福建福清"
    ]
  },
  {
    "question": "侯建国什么时候出生",
    "answers": [
      "1959年10月"
    ]
  },
  {
    "question": "万立骏是哪里人",
    "answers": [
      "辽宁大连"
    ]
  },
  {
    "question": "万立骏什么时候出生",
    "answers": [
      "1957年7月"
    ]
  },
  {
    "question": "谁是科大首任数学系主任",
    "answers": [
      "华罗庚"
    ]
  },
  {
    "question": "华罗庚是哪里人",
    "answers": [
      "江苏丹阳"
    ]
  },
  {
    "question": "华罗庚什么时候出生",
    "answers": [
      "1910年11月"
    ]
  },
  {
    "question": "谁是科大首任近代力学系主任",
    "answers": [
      "钱学森"
    ]
  },
  {
    "question": "钱学森是哪里人",
    "answers": [
      "浙江杭州"
    ]
  },
  {
    "question": "钱学森什么时候出生",
    "answers": [
      "1911年12月"
    ]
  },
  {
    "question": "谁是科大首任物理化学系主任",
    "answers": [
      "郭永怀"
    ]
  },
  {
    "question": "郭永怀是哪里人",
    "answers": [
      "山东荣成"
    ]
  },
  {
    "question": "郭永怀什么时候出生",
    "answers": [
      "1909年4月"
    ]
  },
  {
    "question": "谁是中科大现任信息科学技术学院院长",
    "answers": [
      "吴枫"
    ]
  },
  {
    "question": "谁是中科大现任少年班学院院长",
    "answers": [
      "陈旸"
    ]
  },
  {
    "question": "中科大什么时候50周年校庆",
    "answers": [
      "2008年9月"
    ]
  },
  {
    "question": "中科大什么时候创建的少年班",
    "answers": [
      "1978年3月"
    ]
  },
  {
    "question": "庄小威是哪里人",
    "answers": [
      "江苏省如皋市"
    ]
  },
  {
    "question": "庄小威什么时候出生的",
    "answers": [
      "1972年1月"
    ]
  },
  {
    "question": "中科大是什么时候搬到合肥的",
    "answers": [
      "1970年10月"
    ]
  },
  {
    "question": "谁是科大首任生物物理系主任",
    "answers": [
      "贝时璋"
    ]
  },
  {
    "question": "贝时璋是哪里人",
    "answers": [
      "浙江镇海"
    ]
  },
  {
    "question": "贝时璋什么时候出生的",
    "answers": [
      "1903年10月"
    ]
  }
]
'''

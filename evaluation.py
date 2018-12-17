"""
evaluation.py
~~~~~~~~~~~~~

Evaluates the performance of the `mini_qa.py` question-answering
system.
"""

#### Library imports
from __future__ import division
import mini_qa

# Standard library
import json


class QAPair():
    """
    Stores a question and a list of acceptable answers.
    """
    def __init__(self, question, answers):
        self.question = question
        self.answers = answers


def main():
    evaluate("google")
    evaluate("wolfram")
    evaluate("hybrid")

def evaluate(source="google"):
    """
    Evaluate the question-answering system based on `source`.  Allowed
    values for `source` are explain in the doc string for mini_qa.qa.
    """
    print("Evaluating the question-answering system based on %s" % source)
    qa_pairs = load_qa_pairs()
    num_questions = len(qa_pairs)
    print "Generating candidate answers for %s questions" % num_questions
    perfect_answers = 0
    if source=="google":
        okay_answers = 0
        rank_sum = 0
    if source=="wolfram" or source=="hybrid":
        num_answers = 0
    for (j, qa_pair) in enumerate(qa_pairs):
        print "Processing question %s" % j
        if source=="google":
            cr = correct_results(answers(qa_pair.question), qa_pair.answers)
            if 0 in cr:
                perfect_answers += 1
            if len(cr) > 0:
                okay_answers += 1
                rank_sum += cr[0]
        else: # assume source=="wolfram" or source=="hybrid"
            answer = mini_qa.qa(qa_pair.question, source)
            if answer in qa_pair.answers:
                perfect_answers += 1
            if answer: # answer is not null
                num_answers += 1
    print "{} returned a perfect answer ({:2%})".format(
        perfect_answers, perfect_answers / num_questions)
    if source=="google":
        print "{} of {} had a correct answer in the top 20 ({:.2%})".format(
            okay_answers, num_questions, okay_answers / num_questions)
        print "Average rank for answers in the top 20: {:.2f}".format(
            rank_sum / okay_answers)
    if source=="wolfram" or source=="hybrid":
        print "{} of {} returned a non-null answer".format(
            num_answers, num_questions)

def load_qa_pairs():
    """
    Return a list of QAPair instances, loaded from the file
    qa_pairs.json.
    """
    f = open("qa_pairs.json")
    qa_pairs = json.load(f)
    f.close()
    return [QAPair(qa_pair["question"], qa_pair["answers"]) 
            for qa_pair in qa_pairs]

def answers(question):
    """
    Return a list of the top 20 answers generated by `mini_qa.qa` to
    `question`.
    """
    return [answer for (answer, score) in mini_qa.qa(question)[:20]]

def correct_results(candidate_answers, acceptable_answers):
    """
    Return a list containing the ranking of all elements of the list
    `candidate_answers` which are also in the list of
    `acceptable_answers`.
    """
    return [j for (j, answer) in enumerate(candidate_answers)
            if answer in acceptable_answers]

if __name__ == "__main__":
    main()

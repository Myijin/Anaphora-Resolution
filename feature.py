# coding=utf-8
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.tokenize import sent_tokenize, word_tokenize

class Dependencies:
    def __init__(self, Text, nom, offset, pronoun, paral):
        self.sents = sent_tokenize(Text)
        self.nom = nom
        self.offset = offset
        self.pronoun = pronoun
        self.para = paral

    def findpindex(self):
        """Find index of sentence containing pronoun.
        arguments:
            sents: list of sentences sent_tokenize(text).
            offset: offsets in df['Pronoun-offset'].
        returns:
            (int): index of sentence.
        """
        lens = [len(i) for i in self.sents]
        suml = 0
        for ind, i in enumerate(lens):
            suml += i
            if suml > self.offset:
                break
        return ind

    def findepends(self):
        """Extract features according to Lappin and Leass’ Head noun emphasis,
        Subject emphasis, Accusative emphasis, Indirect object/oblique comp emphasis.
        arguments:
            Text: sentences in df['Text'].
            nom: entity ('Subjectone', 'Subjecttwo').
            offset: offsets in df['Pronoun-offset'].
        returns:
            pd.Series(list(float)):
                list[0]: Head noun emphasis.
                list[1]: Subject emphasis.
                list[2]: Accusative emphasis (direct object).
                list[3]: Indirect object/oblique comp emphasis (object of preposition).
                list[4]: Indirect object/oblique comp emphasis (possession modifier).
                list[5]: grammatical role parallelism.
                list[6]: adverbial.
                list[7]: Non-adverbial emphasis.

        """

        def get_dist(para, x):
            if para > 0:
                y = np.exp(-para * (x * x))
            else:
                y = np.exp(para * x)
            return y

        def count_adverbial(doc, nom):
            """Compute counts of entity that are in prepositional phrase,
            according to Lappin and Leass’ Non-adverbial emphasis."""
            count_pp = 0
            last_pp = ''
            for token in doc:
                if token.pos_ == 'ADP':
                    pp = ' '.join([tok.orth_ for tok in token.subtree])
                    # make sure pp is not part of last pp
                    # e.g. 'in a recent biography by Subjectone' has sub pp 'by Subjectone'
                    if (not last_pp) or (pp not in last_pp):
                        count_pp += pp.count(nom)
                        last_pp = pp
            return count_pp

        indp = self.findpindex()
        lis = [0] * 8
        for ind, i in enumerate(self.sents):
            dist = get_dist(self.para, abs(indp - ind))
            doc = nlp(unicode(i))

            lisp = []
            if ind == indp:
                for token in doc:
                    if token.text == self.pronoun:
                        lisp.append(token.dep_)

            # check if in head noun
            lis_hnp = []
            for ichunk in doc.noun_chunks:
                lis_hnp.append(str(ichunk[-1]))
            if self.nom in lis_hnp:
                lis[0] += dist

            # check if the word is subject, object, possessive, etc.
            for token in doc:
                if token.text == self.nom:
                    if token.dep_ in lisp:
                        lis[5] += dist
                    if ('subj' in token.dep_ or token.dep_ == 'ROOT'):
                        lis[1] += dist
                    elif token.dep_ == 'dobj':
                        lis[2] += dist
                    elif token.dep_ in ['pobj', 'conj']:
                        lis[3] += dist
                    elif token.dep_ == 'poss':
                        lis[4] += dist

            # count entity that are not in prepositional phrase
            count_pp = count_adverbial(doc, self.nom)
            count_npp = i.count(self.nom) - count_pp
            if count_pp > 0:
                lis[6] += dist
            if count_npp > 0:
                lis[7] += dist
        return pd.Series(lis)


def pronounr(Pronoun):
    """Return type of pronoun."""
    if Pronoun in ['He', 'She', 'he', 'she']:
        return 2
    if Pronoun in ['his', 'her', 'hers']:
        return 1
    return 0


def positionw(Pronoun_offset, w, sentence):
    """Return counts of entity mentioned before or after pronoun,
    according to Lappin and Leass’ Existential emphasis."""
    pred = sentence[:Pronoun_offset].count(w)
    after = sentence[Pronoun_offset + 3:].count(w)
    return pd.Series([pred, after])


def possentence(Text, nom, Pronoun):
    """Find if entity is mentioned in sentences mentioning pronoun,
    according to Lappin and Leass’ Sentence recency.
    arguments:
        Text: sentences in df['Text'].
        nom: entity ('Subjectone', 'Subjecttwo').
        Pronoun: pronoun in df['Pronoun']
    returns:
        pd.Series(list(int)):
            list[0]: 1 if entity and pronoun are in the same sentence.
            list[1]: 1 if entity and pronoun are in the same sentence and entity appears after pronoun.
            list[2]: 1 if entity appears in the sentence before.
    """
    out_list = []
    sents = sent_tokenize(Text)
    sent_tokens = [word_tokenize(i) for i in sents]
    for words in sent_tokens:
        index_pronoun = list(np.where(np.array(words) == Pronoun)[0])
        index_nom = [i for i, s in enumerate(words) if nom in s]
        if index_pronoun and index_nom:
            if index_nom[-1] > index_pronoun[0]:
                out_list.append(3)  # in the same sentence and entity appears before pronoun
            else:
                out_list.append(2)  # in the same sentence and entity appears after pronoun
    A_ind = [ind for ind, i in enumerate(sents) if nom in i]
    p_ind = [ind for ind, i in enumerate(sent_tokens) if Pronoun in i]
    p_ind = [i - 1 for i in p_ind if i > 0]
    if any(i in p_ind for i in A_ind):
        out_list.append(1)  # two sent

    if not out_list:
        return 0
    return max(out_list)

def get_features(df, para):
    df['pronoun_type'] = df.apply(lambda r: pronounr(r['Pronoun']), axis=1)
    df[['pred_A', 'after_A']] = df.apply(lambda r: positionw(r['Pronoun-offset'], r['A_Noun'], r['Text']), axis=1)
    df[['pred_B', 'after_B']] = df.apply(lambda r: positionw(r['Pronoun-offset'], r['B_Noun'], r['Text']), axis=1)
    df['Text_sub'] = df.apply(lambda r: r['Text'].replace(r['A_Noun'], 'Subjectone'), axis=1)
    df['Text_sub'] = df.apply(lambda r: r['Text_sub'].replace(r['B_Noun'], 'Subjecttwo'), axis=1)
    df[['head_A', 'nsubj_A', 'dobj_A', 'pobj_A', 'poss_A', 'paral_A', 'ad_A', 'nonad_A']] = df.apply(
        lambda row: Dependencies(row['Text_sub'], 'Subjectone', row['Pronoun-offset'], row['Pronoun'], para).findepends(), axis=1)
    df[['head_B', 'nsubj_B', 'dobj_B', 'pobj_B', 'poss_B', 'paral_B', 'ad_B', 'nonad_B']] = df.apply(
        lambda row: Dependencies(row['Text_sub'], 'Subjecttwo', row['Pronoun-offset'], row['Pronoun'], para).findepends(), axis=1)
    df['pos_sent_A'] = df.apply(lambda row: possentence(row['Text_sub'], 'Subjectone', row['Pronoun']), axis=1)
    df['pos_sent_B'] = df.apply(lambda row: possentence(row['Text_sub'], 'Subjecttwo', row['Pronoun']), axis=1)
    df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()
    df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()
    return df
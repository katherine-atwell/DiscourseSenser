#!/usr/bin/env python
# -*- coding: utf-8; -*-

"""The Official CONLL 2016 Shared Task Scorer

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from collections import defaultdict
from confusion_matrix import ConfusionMatrix, Alphabet
from conn_head_mapper import ConnHeadMapper
import validator

import argparse
import json
import sys
import time

##################################################################
# Constants
CONN_HEAD_MAPPER = ConnHeadMapper()
ENCODING = 'utf-8'


##################################################################
# Methods
def evaluate(gold_list, predicted_list, verbose=False):
    connective_cm = evaluate_connectives(gold_list, predicted_list, verbose)
    arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list,
                                                               predicted_list,
                                                               verbose)
    sense_cm = evaluate_sense(gold_list, predicted_list, verbose)

    print('Explicit connectives         : Precision %1.4f Recall %1.4f'
          ' F1 %1.4f' % connective_cm.get_prf('yes'))
    print('Arg 1 extractor              : Precision %1.4f Recall %1.4f'
          ' F1 %1.4f' % arg1_cm.get_prf('yes'))
    print('Arg 2 extractor              : Precision %1.4f Recall %1.4f'
          ' F1 %1.4f' % arg2_cm.get_prf('yes'))
    print('Arg1 Arg2 extractor combined : Precision %1.4f Recall %1.4f'
          ' F1 %1.4f' % rel_arg_cm.get_prf('yes'))
    print('Sense classification--------------')
    sense_cm.print_summary()
    print('Overall parser performance --------------')
    precision, recall, f1 = sense_cm.compute_micro_average_f1()
    print('Precision %1.4f Recall %1.4f F1 %1.4f' % (precision, recall, f1))
    # sense_cm.plot()
    return (connective_cm, arg1_cm, arg2_cm, rel_arg_cm,
            sense_cm, precision, recall, f1)


def evaluate_argument_extractor(gold_list, predicted_list, verbose=False):
    """Evaluate argument extractor at Arg1, Arg2, and relation level

    """
    gold_arg1 = [(x['DocID'], tuple(t[2] for t in x['Arg1']['TokenList']))
                 for x in gold_list]
    predicted_arg1 = [(x['DocID'], tuple(x['Arg1']['TokenList']))
                      for x in predicted_list]
    arg1_cm = compute_span_exact_match_metric(gold_arg1, predicted_arg1)

    gold_arg2 = [(x['DocID'], tuple(t[2] for t in x['Arg2']['TokenList']))
                 for x in gold_list]
    predicted_arg2 = [(x['DocID'], tuple(x['Arg2']['TokenList']))
                      for x in predicted_list]
    arg2_cm = compute_span_exact_match_metric(gold_arg2, predicted_arg2)

    gold_arg12 = [(g1[0], (g1[-1], g2[-1]))
                  for g1, g2 in zip(gold_arg1, gold_arg2)]
    predicted_arg12 = [(p1[0], (p1[-1], p2[-1]))
                       for p1, p2 in zip(predicted_arg1, predicted_arg2)]
    rel_arg_cm = compute_span_exact_match_metric(gold_arg12, predicted_arg12,
                                                 verbose)
    return arg1_cm, arg2_cm, rel_arg_cm


def evaluate_connectives(gold_list, predicted_list, verbose=False):
    """Evaluate connective accuracy for explicit discourse relations

    """
    explicit_gold_list = [(x['DocID'],
                           set(t[2] for t in x['Connective']['TokenList']),
                           [t[2] for t in x['Connective']['TokenList']],
                           x['Connective']['RawText'])
                          for x in gold_list if x['Type'] == 'Explicit']
    explicit_predicted_list = [(x['DocID'], set(x['Connective']['TokenList']))
                               for x in predicted_list
                               if x['Type'] == 'Explicit']
    connective_cm = compute_binary_eval_metric(
        explicit_gold_list, explicit_predicted_list,
        connective_head_matching)
    return connective_cm


def spans_exact_matching(gold_doc_id_spans, predicted_doc_id_spans):
    """Matching two lists of spans

    Input:
        gold_doc_id_spans : (DocID , a list of lists of tuples of token
        addresses)
        predicted_doc_id_spans : (DocID , a list of lists of token indices)

    Returns:
        True if the spans match exactly

    """
    gold_docID = gold_doc_id_spans[0]
    predicted_docID = predicted_doc_id_spans[0]

    for gold_span, predicted_span in zip(gold_doc_id_spans[1],
                                         predicted_doc_id_spans[1]):
        if not span_exact_matching((gold_docID, gold_span),
                                   (predicted_docID, predicted_span)):
            return False
    return True


def span_exact_matching(gold_span, predicted_span):
    """Matching two spans

    Input:
        gold_span : a list of tuples :(DocID, list of tuples of token
        addresses)
        predicted_span : a list of tuples :(DocID, list of token indices)

    Returns:
        True if the spans match exactly

    """
    if gold_span[0] != predicted_span[0] or \
       len(gold_span[1]) != len(predicted_span[1]):
        return False
    for x1, x2 in zip(gold_span[1], predicted_span[1]):
        if x1[2] != x2:
            return False
    return True


def connective_head_matching(gold_raw_connective, predicted_raw_connective):
    """Matching connectives

    Input:
        gold_raw_connective : (DocID, a list of tuples of token addresses, raw
        connective token)
        predicted_raw_connective : (DocID, a list of tuples of token addresses)

    A predicted raw connective is considered iff
        1) the predicted raw connective includes the connective "head"
        2) the predicted raw connective tokens are the subset of predicted raw
        connective tokens

    For example:
        connective_head_matching('two weeks after', 'weeks after')  --> True
        connective_head_matching('two weeks after', 'two weeks') --> False not
        covering head
        connective_head_matching('just because', 'because')  --> True
        connective_head_matching('just because', 'simply because') --> False
        not subset
        connective_head_matching('just because', 'since')  --> False

    """
    gold_docID, gold_token_indices, gold_token_list, \
        gold_tokens = gold_raw_connective
    predicted_docID, predicted_token_indices = predicted_raw_connective
    if gold_docID != predicted_docID:
        return False

    if gold_token_indices == predicted_token_indices:
        return True
    elif not predicted_token_indices.issubset(gold_token_indices):
        return False
    else:
        conn_head, indices = CONN_HEAD_MAPPER.map_raw_connective(gold_tokens)
        for x in indices:
            if gold_token_list[x] not in predicted_token_indices:
                return False
        return True


def evaluate_sense(gold_list, predicted_list, verbose=False):
    """Evaluate sense classifier

    The label ConfusionMatrix.NEGATIVE_CLASS is for the relations

    that are missed by the system
    because the arguments don't match any of the gold relations.

    """
    sense_alphabet = Alphabet()
    valid_senses = validator.identify_valid_senses(gold_list)

    isense = None
    for relation in gold_list:
        isense = relation['Sense'][0]
        if isense in valid_senses:
            sense_alphabet.add(isense)

    sense_alphabet.add(ConfusionMatrix.NEGATIVE_CLASS)

    sense_cm = ConfusionMatrix(sense_alphabet)
    gold_to_predicted_map, predicted_to_gold_map = \
        _link_gold_predicted(gold_list, predicted_list,
                             spans_exact_matching)

    for i, gold_relation in enumerate(gold_list):
        gold_sense = gold_relation['Sense'][0]
        if gold_sense in valid_senses:
            if i in gold_to_predicted_map:
                predicted_sense = gold_to_predicted_map[i]['Sense'][0]
                if predicted_sense in gold_relation['Sense']:
                    sense_cm.add(predicted_sense, predicted_sense)
                else:
                    if not sense_cm.alphabet.has_label(predicted_sense):
                        predicted_sense = ConfusionMatrix.NEGATIVE_CLASS
                    if verbose:
                        print('Sense:')
                        print('<<<\t{:s}'.format(gold_sense).encode(ENCODING))
                        print('>>>\t{:s}'.format(predicted_sense).encode(
                            ENCODING))
                        print('Arg1:\t{:s}'.format(
                            gold_relation['Arg1']['RawText']).encode(ENCODING))
                        print('Arg2:\t{:s}'.format(
                            gold_relation['Arg2']['RawText']).encode(ENCODING))
                        print()
                    sense_cm.add(predicted_sense, gold_sense)
            else:
                if verbose:
                    print('Sense:')
                    print('<<<\t{:s}'.format(gold_sense).encode(ENCODING))
                    print('>>>\t{:s}'.format(
                        ConfusionMatrix.NEGATIVE_CLASS).encode(
                        ENCODING))
                    print('Arg1:\t{:s}'.format(
                        gold_relation['Arg1']['RawText']).encode(ENCODING))
                    print('Arg2:\t{:s}'.format(
                        gold_relation['Arg2']['RawText']).encode(ENCODING))
                    print()
                sense_cm.add(ConfusionMatrix.NEGATIVE_CLASS, gold_sense)

    for i, predicted_relation in enumerate(predicted_list):
        if i not in predicted_to_gold_map:
            predicted_sense = predicted_relation['Sense'][0]
            if not sense_cm.alphabet.has_label(predicted_sense):
                predicted_sense = ConfusionMatrix.NEGATIVE_CLASS
            if verbose:
                print('Sense:')
                print('<<<\t{:s}'.format(gold_sense).encode(ENCODING))
                print('>>>\t{:s}'.format(
                    ConfusionMatrix.NEGATIVE_CLASS).encode(
                    ENCODING))
                print('Arg1:\t{:s}'.format(
                    gold_relation['Arg1']['RawText']).encode(ENCODING))
                print('Arg2:\t{:s}'.format(
                    gold_relation['Arg2']['RawText']).encode(ENCODING))
                print()
            sense_cm.add(predicted_sense, ConfusionMatrix.NEGATIVE_CLASS)
    return sense_cm


def combine_spans(span1, span2):
    """Merge two text span dictionaries

    """
    new_span = {}
    new_span['CharacterSpanList'] = span1['CharacterSpanList'] + \
        span2['CharacterSpanList']
    new_span['SpanList'] = span1['SpanList'] + span2['SpanList']
    new_span['RawText'] = span1['RawText'] + span2['RawText']
    new_span['TokenList'] = span1['TokenList'] + span2['TokenList']
    return new_span


def compute_span_exact_match_metric(gold_list, predicted_list, verbose=False):
    """Compute binary evaluation metric

    """
    binary_alphabet = Alphabet()
    binary_alphabet.add('yes')
    binary_alphabet.add('no')
    cm = ConfusionMatrix(binary_alphabet)
    matched_predicted = [False for x in predicted_list]
    predicted = defaultdict(list)
    for i, pspan in enumerate(predicted_list):
        predicted[pspan].append(i)
    empty_list = []
    key = indices = None
    for gold in gold_list:
        found_match = False
        indices = predicted.get(gold, empty_list)
        for i in indices:
            if not matched_predicted[i]:
                cm.add('yes', 'yes')
                matched_predicted[i] = True
                found_match = True
                break
        if not found_match:
            if verbose:
                print('Span:')
                print('<<<\t{:s}'.format(gold).encode(ENCODING))
                print()
            cm.add('no', 'yes')
    # Predicted span that does not match with any
    for matched, pred in zip(matched_predicted, predicted_list):
        if not matched:
            if verbose:
                print('Span:')
                print('>>>\t{:s}'.format(pred).encode(ENCODING))
                print()
            cm.add('yes', 'no')
    return cm


def compute_binary_eval_metric(gold_list, predicted_list, matching_fn):
    """Compute binary evaluation metric

    """
    binary_alphabet = Alphabet()
    binary_alphabet.add('yes')
    binary_alphabet.add('no')
    cm = ConfusionMatrix(binary_alphabet)
    matched_predicted = [False for x in predicted_list]
    for gold_span in gold_list:
        found_match = False
        for i, predicted_span in enumerate(predicted_list):
            if matching_fn(gold_span, predicted_span) and \
               not matched_predicted[i]:
                cm.add('yes', 'yes')
                matched_predicted[i] = True
                found_match = True
                break
        if not found_match:
            cm.add('no', 'yes')
    # Predicted span that does not match with any
    for matched in matched_predicted:
        if not matched:
            cm.add('yes', 'no')
    return cm


def _link_gold_predicted(gold_list, predicted_list, matching_fn):
    """Link gold standard relations to the predicted relations

    A pair of relations are linked when the arg1 and the arg2 match exactly.
    We do this because we want to evaluate sense classification later.

    Returns:
        A tuple of two dictionaries:
        1) mapping from gold relation index to predicted relation index
        2) mapping from predicted relation index to gold relation index
    """
    gold_to_predicted_map = {}
    predicted_to_gold_map = {}
    gold_arg12_list = [(x['DocID'],
                        (tuple(t[2] for t in x['Arg1']['TokenList']),
                         tuple(t[2] for t in x['Arg2']['TokenList'])))
                       for x in gold_list]
    predicted_arg12_list = [(x['DocID'], (tuple(x['Arg1']['TokenList']),
                                          tuple(x['Arg2']['TokenList'])))
                            for x in predicted_list]
    predictions = {k: i for i, k in enumerate(predicted_arg12_list)}
    pi = -1
    for gi, gold_span in enumerate(gold_arg12_list):
        if gold_span in predictions:
            pi = predictions[gold_span]
            gold_to_predicted_map[gi] = predicted_list[pi]
            predicted_to_gold_map[pi] = gold_list[gi]
    return gold_to_predicted_map, predicted_to_gold_map


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate system's output against the gold standard")
    parser.add_argument('-v', '--verbose',
                        help='Output erroneous predictions.',
                        action='store_true')
    parser.add_argument('gold', help='Gold standard file')
    parser.add_argument('predicted', help='System output file')
    args = parser.parse_args()
    gold_list = [json.loads(x) for x in open(args.gold) if x.strip()]
    predicted_list = [json.loads(x) for x in open(args.predicted) if x.strip()]

    print('\n================================================')
    print('Evaluation for all discourse relations')
    evaluate(gold_list, predicted_list, args.verbose)

    print('\n================================================')
    print('Evaluation for explicit discourse relations only')
    explicit_gold_list = [x for x in gold_list if x['Type'] == 'Explicit']
    explicit_predicted_list = [x for x in predicted_list
                               if x['Type'] == 'Explicit']
    evaluate(explicit_gold_list, explicit_predicted_list)

    print('\n================================================')
    print('Evaluation for non-explicit discourse relations only'
          ' (Implicit, EntRel, AltLex)')
    non_explicit_gold_list = [x for x in gold_list
                              if x['Type'] != 'Explicit']
    non_explicit_predicted_list = [x for x in predicted_list
                                   if x['Type'] != 'Explicit']
    evaluate(non_explicit_gold_list, non_explicit_predicted_list)

##################################################################
# Main
if __name__ == '__main__':
    main()

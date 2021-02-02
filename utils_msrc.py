#opyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open

from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

#added
from nltk.tokenize import sent_tokenize
#end

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_msrc_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores
from utils_q_type import *
logger = logging.getLogger(__name__)

#changed
import nltk
IdentifyTags=['JJR','JJS','RBR','RBS']#adj comparative/superlative ;adv comparative/superlative ;
IdentifyWords=['most','last','first','more','second','third']

def new_word_add_sent_logits(word_logits,sent_logits,toke_to_eos):
    id = sent_logits.index(max(sent_logits))
    w_s_logits = [0]*len(toke_to_eos)
    if id !=2:
        for i in range(len(toke_to_eos)):
            if toke_to_eos[i] == id:
                w_s_logits[i] = word_logits[i] + sent_logits[toke_to_eos[i]]
            else:
                w_s_logits[i] = -100000
    else:
        for i in range(len(toke_to_eos)):
            w_s_logits[i] = word_logits[i] + sent_logits[toke_to_eos[i]]
    return w_s_logits

def sent_logits_for_wordlevel(sent_logits,tok_to_eos):
    #add a wordlevel logit and sent_logits
    sent_logit_for_w = [0]*len(tok_to_eos)
    for i in range(len(tok_to_eos)):
        if tok_to_eos[i] == 0:
            sent_logit_for_w[i] = -9999
        else:
            sent_logit_for_w[i] = sent_logits[tok_to_eos[i]]
    return sent_logit_for_w
def word_add_sent_logits(word_logits,sent_logits,toke_to_eos):
    w_s_logits = [0]*len(toke_to_eos)
    for i in range(len(toke_to_eos)):
        w_s_logits[i] = word_logits[i] + sent_logits[toke_to_eos[i]]
    return w_s_logits
#end
def CMP_question(question_text):
    tagged_text = nltk.pos_tag(nltk.word_tokenize(question_text))
    Tag=False
    for word, tag in tagged_text:
        if (tag in IdentifyTags) or (word in IdentifyWords):
            Tag=True
    return Tag
#end
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 q_type=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.q_type=q_type

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 sent_segs,#added2
                 tok_to_eos,#added3
                 w_mask,#added5
                 s_mask,#added6
                 query_attention_mask=None,
                 q_input_ids=None,#added 4.1
                 q_input_mask=None,#added 4.2
                 q_segment_ids=None,#added 4.3
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 sent_position=None,#added1
                 para_label=None,#added7
                 q_type=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.sent_position=sent_position#added1
        self.sent_segs = sent_segs#added2
        self.tok_to_eos = tok_to_eos#added3
        self.q_input_ids = q_input_ids#added4.1
        self.q_input_mask = q_input_mask#added4.2
        self.q_segment_ids = q_segment_ids#added4.3
        self.w_mask = w_mask#added 5
        self.s_mask = s_mask#added6
        self.para_label=para_label#added7
        self.q_type=q_type
        self.query_attention_mask=query_attention_mask



def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def my_sent_tag(para_text):
        char_segs_for_eos = []
        for i in range(len(para_text) - 1):
            if para_text[i] == "." or para_text[i] == "?" or para_text[i] == "!":
                if is_whitespace(para_text[i + 1]):
                    if i>=4:
                        if (not is_whitespace(para_text[i - 1])) and (not is_whitespace(para_text[i - 2])) and (not is_whitespace(para_text[i - 3])) and (not is_whitespace(para_text[i-4])):
                            char_segs_for_eos.append(i + 1)
        return char_segs_for_eos
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            char_segs_for_eos = my_sent_tag(paragraph_text)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)#please note that the len of char_to_word_offset is equal to the original text
                if len(char_to_word_offset) in char_segs_for_eos:
                    doc_tokens.append("[EOS]")
                    prev_is_whitespace = True
            if doc_tokens[-1]!="[EOS]":
                doc_tokens.append("[EOS]")

            assert len(char_to_word_offset)==len(paragraph_text)#lqq added
            # if len(char_to_word_offset)!=len(paragraph_text):
            #     logger.warning("char level is not match: '%s' vs. '%s'",
            #                    len(char_to_word_offset), len(paragraph_text))
            #     print("-----------")
            #     print(len(char_segs_for_eos),len(paragraph_text))
            #     print(doc_tokens)
            #     print(paragraph_text)
            #     print(char_to_word_offset)
            #     continue
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                q_type = getQTag(question_text)[0]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    # if (len(qa["answers"]) != 1) and (not is_impossible):
                    #     raise ValueError(
                    #         "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]#lq changed
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]#lq changed
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        # actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue

                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    q_type=q_type,
                    question_text=question_text,
                    doc_tokens=doc_tokens,#doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,isMergedSen,num_tags,num_berts,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""
    ##verfied
    if num_berts==1 or num_berts==2:
        print("**************used berts:{}".format(num_berts))
    else:
        raise ValueError(
            "The parameter num_berts is wrong. It is expected to be 1 or 2.")
    if num_tags==1 or num_tags==3:
        print("**************used tags:{}".format(num_tags))
    else:
        raise ValueError(
            "The parameter num_tags is wrong. It is expected to be 1 or 3.")
    print("**************is merged sent:{}".format(isMergedSen))
    #end

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(examples):

        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        #query tokens
        q_seq_length = max_query_length+6
        query_tokens=[]#used for first bert.
        q_token = []#used for second bert, if num_bert ==1 use the []
        q_input_ids=[]
        q_segment_ids = []
        q_input_mask = []
        query_attention_mask = []
        if num_berts == 2:
            query_tokens.extend(tokenizer.tokenize(example.question_text))
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]
            # for another bert
            if num_tags == 3:
                q_token.extend(["[CLS]","[SEM]", "[CMP]", "[TPE]", "[SEP]"])
                q_segment_ids.extend([0,0,0,0,0])
                query_attention_mask.extend([0,0,0,0,0])
            if num_tags == 1:
                q_token.extend(["[CLS]","[CMP]",  "[SEP]"])
                q_segment_ids.extend([0,0,0])
                query_attention_mask.extend([0,0,0])

            tokenized_query = tokenizer.tokenize(example.question_text)
            if len(tokenized_query) > max_query_length:
                tokenized_query = tokenized_query[0:max_query_length]
            q_token.extend(tokenized_query)
            q_segment_ids.extend([1] * len(tokenized_query))
            query_attention_mask.extend([1]*len(tokenized_query))
            q_token.extend(["[SEP]"])
            q_segment_ids.extend([1])
            query_attention_mask.extend([0])#!!!different
            q_input_ids = tokenizer.convert_tokens_to_ids(q_token)
            q_input_mask = [1] * len(q_token)
            q_padding = [0] * (q_seq_length - len(q_token))#old:max_seq_length
            q_segment_ids.extend(q_padding)
            q_input_mask.extend(q_padding)
            q_input_ids.extend(q_padding)
            query_attention_mask.extend(q_padding)
            assert len(q_segment_ids) == q_seq_length
            assert len(q_input_mask) == q_seq_length
            assert len(q_input_ids) == q_seq_length
            assert len(query_attention_mask) == q_seq_length


        if num_berts == 1 and num_tags == 1:
            query_tokens.append("[CMP]")
            query_tokens.extend(tokenizer.tokenize(example.question_text))
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]
            q_input_ids = None
            q_segment_ids = None
            q_input_mask = None
            query_attention_mask.extend([0,0,0,0])#CLS PCLS SCLS CMP question sep doc sep
            query_attention_mask.extend([1]*len(query_tokens))
            temp_pad = max_seq_length-len(query_attention_mask)
            query_attention_mask.extend([0]*(temp_pad))

        if num_berts == 1 and num_tags ==3:
            query_tokens.extend(["[SEM]","[CMP]","[TPE]"])
            query_tokens.extend(tokenizer.tokenize(example.question_text))
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]
            q_input_ids = None
            q_segment_ids = None
            q_input_mask = None
            query_attention_mask.extend([0, 0, 0, 0,0,0])  # CLS PCLS SCLS SEM CMP TPE question sep doc sep
            query_attention_mask.extend([1] * len(query_tokens))
            temp_pad = max_seq_length - len(query_attention_mask)
            query_attention_mask.extend([0] * (temp_pad))


        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP] #modified 3 to 4 another [EOS] before the last [SEP]
        # modified 4 to 6, since we added two more tags:[PCLS] [SCLS]
        # max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # max_tokens_for_doc = max_seq_length - len(query_tokens) - 4
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 6

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # added:For sent answer
        all_sent_map = []#
        for my_iter in range(len(tok_to_orig_index)):
            orig_idx = tok_to_orig_index[my_iter]  # the idx in example.doc_tokens(tok)/example.doc_tokens_map(tok_id in html)
            tok = example.doc_tokens[orig_idx]
            if tok == '[EOS]':
                all_sent_map.append(1)
            else:
                all_sent_map.append(0)
        tok_sent_position = None
        if is_training and example.is_impossible:
            tok_sent_position = -1
        if is_training and not example.is_impossible:
            pre_sent_temp = 0
            for my_iter in range(len(all_sent_map)):
                if all_sent_map[my_iter]:
                    if pre_sent_temp <= tok_start_position and my_iter > tok_start_position:#here is <= for the cases that start is 0
                        tok_sent_position = my_iter
                        break
                    else:
                        pre_sent_temp = my_iter

        ############################################start the doc chunk###############
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []
            #s_mask is added  , if 1 it is a eos, elso other token. [cls] is 1. the logic is different with p_mask
            s_mask = []
            #w_mask is added , if 1 it is a word indoc, [cls] is 1.
            w_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
                s_mask.append(0)
                w_mask.append(1)
            #added the [PCLS] and [SCLS]
            tokens.append("[PCLS]")
            segment_ids.append(0)
            p_mask.append(0)
            s_mask.append(0)
            w_mask.append(0)

            tokens.append("[SCLS]")
            segment_ids.append(0)
            p_mask.append(0)
            s_mask.append(1)
            w_mask.append(0)

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
                s_mask.append(0)
                w_mask.append(0)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)
            s_mask.append(0)
            w_mask.append(0)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
                s_mask.append(all_sent_map[split_token_index])#lqq added.
                w_mask.append(1-all_sent_map[split_token_index])#all sent map: eos is 1 others is 0;since there is the loop of doc tokens. thus, 1-
            paragraph_len = doc_span.length

            # added the last [EOS]
            tokens.append('[EOS]')
            segment_ids.append(sequence_b_segment_id)
            insurance_eos= len(tokens)-1##know the added eos's id
            p_mask.append(1)
            s_mask.append(1)#the added [EOS]
            w_mask.append(0)
            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            s_mask.append(0)
            w_mask.append(0)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                s_mask.append(0)
                w_mask.append(1)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            my_padding = [0] * (max_seq_length - len(input_ids))#lqq added
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)
                s_mask.append(0)
                w_mask.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(w_mask) == max_seq_length
            assert len(s_mask) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            sent_position = None# added
            para_label = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                para_label=1
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                    sent_position = 2#!! added
                else:
                    doc_offset = len(query_tokens) + 4#here changed!we need to add the [CLS][PCLS][SCLS][SEP]
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    # added
                    out_of_s = False#whether this doc_span contain a right [EOS] tag
                    if not (tok_sent_position >= doc_start and tok_sent_position <=doc_end):
                        out_of_s = True
                    if out_of_s:
                        sent_position = insurance_eos
                        print(start_position,end_position,sent_position)
                    else:
                        sent_position = tok_sent_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index
                sent_position = 2#lq added#lq modified to [SCLS]
                para_label=0
            ##Lq added
            token_to_eos = [0] * max_seq_length
            eos_idxs = [i for i, e_i in enumerate(s_mask) if
                        e_i != 0]  # the appeneded [EOS] in 510 is already in this list
            for i in range(0, len(eos_idxs) - 1):
                for j in range(eos_idxs[i] + 1, eos_idxs[i + 1] + 1):
                    token_to_eos[j] = eos_idxs[i + 1]
            for i in range(0, len(query_tokens) + 4):  # the token_to_eos before [SEP] should be [SCLS]
                token_to_eos[i] = 2# modified
            # sent segs
            # sent position
            # sent_pos_ids = []
            max_sents_num=6#here is 6! since we added an extra [EOS] at last position.
            if isMergedSen and len(eos_idxs)>max_sents_num: #version1:sen_segs: merged sentences, and no more than 10 sents
                sent_segs = [0] * (len(query_tokens) + 2)
                sent_merged_size = int(math.ceil((len(eos_idxs) - 1) / max_sents_num))
                for i in range(1, len(eos_idxs)):
                    sent_segs.extend([int(math.ceil(i / sent_merged_size))] * (eos_idxs[i] - len(sent_segs) + 1))
                sent_segs.extend(my_padding)
                sent_segs.append(0)  # the last [sep]
                assert len(sent_segs) == max_seq_length
            else:# use the natural sent segs. #version1:sent_segs is [00000,1111,2222,3333,]
                sent_segs = [0]*(len(query_tokens)+2)
                for i in range(1, len(eos_idxs)):
                    sent_segs.extend([i] * (eos_idxs[i] - len(sent_segs) + 1))
                sent_segs.extend(my_padding)
                sent_segs.append(0)#the last [sep]
                assert len(sent_segs) == max_seq_length

            if example_index<5 and is_training:#example_index < 50
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("orig_passage:%s" % " ".join(example.doc_tokens))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                # #lqq added
                # logger.info("lq_added sent_segs: %s" % " ".join([str(x) for x in sent_segs]))
                # logger.info("lq_added tok_to_eos: %s" % " ".join([str(x) for x in token_to_eos]))
                # logger.info("lq_added w_mask: %s" % " ".join([str(x) for x in w_mask]))
                # logger.info("lq_added s_mask: %s" % " ".join([str(x) for x in s_mask]))
                # if num_berts == 2:
                #     logger.info("lq_added q_token: %s" % " ".join([str(x) for x in q_token]))
                #     logger.info("lq_added q_input_ids: %s" % " ".join([str(x) for x in q_input_ids]))
                #     logger.info("lq_added q_input_mask: %s" % " ".join([str(x) for x in q_input_mask]))
                #     logger.info("lq_added q_segment_ids: %s" % " ".join([str(x) for x in q_segment_ids]))

                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("questions: %s" % (example.question_text))
                    logger.info("questions_type: %d" % (example.q_type))
                    logger.info("query_attenion_mask:%s" % " ".join([str(x) for x in query_attention_mask]))
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("sent_position: %d" % (sent_position))
                    logger.info("para_label: %d" % (para_label))
                    logger.info("orig answer: %s" % (example.orig_answer_text))
                    logger.info("feature answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    q_type=example.q_type,
                    is_impossible=span_is_impossible,
                    sent_position=sent_position,#added1
                    sent_segs=sent_segs,#added2
                    tok_to_eos=token_to_eos,#added3
                    q_input_ids=q_input_ids,#added4
                    q_input_mask=q_input_mask,
                    q_segment_ids=q_segment_ids,
                    w_mask=w_mask,#added5
                    s_mask=s_mask,#added6
                    para_label=para_label,#added7
                    query_attention_mask=query_attention_mask
                ))
            unique_id += 1
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits","para_logits","sent_logits","tpe_logits"])

def write_predictions_psw(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold,para_threshold):
    """consider the para score, sent score, and word score"""
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit","word_add_sent_score","para_score","word_diff","sent_diff"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    has_answer = 0
    phas = 0
    shas = 0
    whas = 0
    swhas = 0
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            para_score = result.para_logits[0]
            sent_score = sent_logits_for_wordlevel(result.sent_logits, feature.tok_to_eos)
            ws_start_logits = new_word_add_sent_logits(result.start_logits, result.sent_logits, feature.tok_to_eos)
            ws_end_logits = new_word_add_sent_logits(result.end_logits, result.sent_logits, feature.tok_to_eos)
            start_indexes = _get_best_indexes(ws_start_logits, n_best_size)
            end_indexes = _get_best_indexes(ws_end_logits, n_best_size)

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            #added
                            word_add_sent_score=result.start_logits[start_index]+result.end_logits[end_index]+sent_score[start_index]+sent_score[end_index],
                            para_score=para_score,
                            word_diff= result.start_logits[0]+result.end_logits[0]-result.start_logits[start_index]-result.end_logits[end_index],
                            sent_diff =sent_score[2]+sent_score[2]-sent_score[start_index]-sent_score[end_index]
                        ))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    word_add_sent_score=0,
                    para_score=0,
                    word_diff=0,
                    sent_diff=0
                ))
        #here sorted!
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.word_add_sent_score),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text","start_logit","end_logit","para_score","sent_diff","word_diff"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    para_score=pred.para_score,
                    word_diff=pred.word_diff,
                    sent_diff=pred.sent_diff
            ))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        para_score=-1,
                        word_diff=0,
                        sent_diff=0
                    ))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(
                        text="empty",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        para_score=-1,
                        word_diff=0,
                        sent_diff=0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="empty",
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    para_score=-1,
                    word_diff=0,
                    sent_diff=0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["para_score"] = entry.para_score
            output["word_diff"]=entry.word_diff
            output["sent_diff"]=entry.sent_diff
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            #word diff cls-word
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            #sent diff
            sent_diff = best_non_null_entry.sent_diff
            #para graph sigmoid
            para_score = best_non_null_entry.para_score
            if para_score>para_threshold and score_diff<0 and sent_diff<0:
                scores_diff_json[example.qas_id] = (score_diff + sent_diff)
                has_answer+=1
                answer = best_non_null_entry.text
                if "[EOS]" in answer.split():
                    all_predictions[example.qas_id] = " ".join(answer.replace("[EOS]", "").split())
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            else:
                scores_diff_json[example.qas_id] = min(1,(score_diff+sent_diff))
                all_predictions[example.qas_id] = ""
            if (score_diff+sent_diff)<0:
                swhas +=1
            if score_diff<0:
                whas +=1
            if sent_diff<0:
                shas +=1
            if para_score>=0.9:
                phas +=1
        all_nbest_json[example.qas_id] = nbest_json
        #answer = best_non_null_entry.text
        #if "[EOS]" in answer.split():
        #    all_predictions[example.qas_id] = " ".join(answer.replace("[EOS]", "").split())
        #else:
        #    all_predictions[example.qas_id] = best_non_null_entry.text


    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    print("phas: {} shas: {} whas: {} sent and word joint has: {}".format(phas,shas,whas,swhas))
    print("finally has answers: {}".format(has_answer))
    return all_predictions

# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                                max_answer_length, output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file, orig_data_file,
                                start_n_top, end_n_top, version_2_with_negative,
                                tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
        "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            #
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs



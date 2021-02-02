from pytorch_transformers.modeling_bert import *
import torch
import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context,mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        # attention_weights = self.softmax(attention_scores)
        attention_weights = masked_softmax(attention_scores,mask)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output#, attention_weights

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
   
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class cmpQA2Bert(BertPreTrainedModel):

    def __init__(self, config, num_tags=None):
        super(cmpQA2Bert, self).__init__(config)

        self.bert = BertModel(config)
        self.apply(self.init_weights)
        self.num_tags = num_tags
        self.num_labels = config.num_labels

        self.tpe_fc = nn.Linear(config.hidden_size,config.hidden_size,bias=True)
        # self.cmp_fc = nn.Linear(config.hidden_size,config.hidden_size,bias=True)
        self.cmp_lstm = nn.LSTM(config.hidden_size,config.hidden_size,num_layers=2)

        self.para_bi = nn.Bilinear(config.hidden_size, config.hidden_size, 1)
        self.sent_bi = nn.Bilinear(config.hidden_size, config.hidden_size, 1)
        self.start_bi = nn.Bilinear(config.hidden_size, config.hidden_size, 1)
        self.end_bi = nn.Bilinear(config.hidden_size, config.hidden_size, 1)
        self.tpe_classifier = nn.Linear(config.hidden_size, 9)

        self.sigmoid = torch.nn.Sigmoid()

        self.attention = Attention(config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, position_ids=None, head_mask=None,
                sent_segs=None,tok_to_eos=None,
                w_mask=None,s_mask=None,query_attention_mask=None,sent_positions=None,para_label=None,
                q_input_ids=None,q_token_type_ids=None,q_attention_mask=None,q_type=None
                ):
        #first bert
        outputs = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        outputs_q = self.bert(q_input_ids, position_ids=None, token_type_ids=q_token_type_ids,
                              attention_mask=q_attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]  # last_hidden_state
        tpe_seq_output = self.tpe_fc(sequence_output)
        self.cmp_lstm.flatten_parameters()# To avoid an warning UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().
        lstm_out, (h, c) = self.cmp_lstm(sequence_output.transpose(1, 0))
        cmp_seq_output = lstm_out.transpose(1,0).contiguous()

        batch_size = sequence_output.shape[0]
        seq_size = sequence_output.shape[1]
        sequence_output_q = outputs_q[0]

        ###tpe_rep with attention

        if self.num_tags == 3:
            sem_tag = sequence_output_q[:, 1, :]  # batch,hidden
            cmp_tag = sequence_output_q[:, 2, :].unsqueeze(1).repeat(1, seq_size, 1)  # batch,seq,hidden
            attented_tpe_tag = self.attention(sequence_output_q[:, 3, :].unsqueeze(1),sequence_output_q,query_attention_mask)
            tpe_tag = attented_tpe_tag.repeat(1, seq_size, 1)  # batch,seq,hidden
            tpe_tag_for_classify = sequence_output[:, 3, :]
        if self.num_tags == 1:
            sem_tag = sequence_output_q[:, 1, :]  # batch,hidden
            cmp_tag = sequence_output_q[:, 1, :].unsqueeze(1).repeat(1, seq_size, 1)  # batch,seq,hidden
            attented_tpe_tag = self.attention(sequence_output_q[:, 1, :].unsqueeze(1), sequence_output_q,
                                              query_attention_mask)
            tpe_tag = attented_tpe_tag.repeat(1, seq_size, 1)  # batch,seq,hidden
            tpe_tag_for_classify = sequence_output[:, 1, :]
        #para

        pcls_rep = sequence_output[:, 1, :]  # batch,hidden[CLS][PCLS][SCLS]question[SEP]doc[SEP]
        para_logits = self.sigmoid(self.para_bi(sem_tag, pcls_rep))
        #sent
        sent_logits = self.sent_bi(cmp_tag, cmp_seq_output).squeeze(-1) + ((s_mask > 0).float() - 1) * 1e10
        #word
        start_logits = self.start_bi(tpe_tag, tpe_seq_output).squeeze(-1) + ((w_mask > 0).float() - 1) * 1e10
        end_logits = self.end_bi(tpe_tag, tpe_seq_output).squeeze(-1) + ((w_mask > 0).float() - 1) * 1e10
        tpe_logits= self.tpe_classifier(tpe_tag_for_classify)

        outputs = (start_logits, end_logits, para_logits,sent_logits, tpe_logits) + outputs[2:]  # tuple
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(sent_positions.size()) > 1:
                sent_positions = sent_positions.squeeze(-1)
            if len(para_label.size()) > 1:
                para_label = para_label.squeeze(-1)
            if len(q_type.size()) >1:
                q_type = para_label.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)  # [batch]
            end_positions.clamp_(0, ignored_index)  # [batch]
            para_label.clamp_(0, ignored_index)
            sent_positions.clamp_(0, ignored_index)  # [batch]
            q_type.clamp_(0, ignored_index)
            #************computing loss***************
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            sent_loss = loss_fct(sent_logits,sent_positions)
            tpe_loss = loss_fct(tpe_logits,q_type)

            loss_fct_para = MSELoss()
            para_loss = loss_fct_para(para_logits.view(-1), para_label.view(-1).float())


            total_loss = (start_loss + end_loss +sent_loss+ tpe_loss+para_loss ) / 5
            outputs = (total_loss,) + outputs  #
            #outputs = (total_loss,) + outputs +(start_loss , end_loss , sent_loss , tpe_loss , para_loss)
# outputs = (start_logits, end_logits, para_logits, sent_logits, tpe_logits) + outputs[2:]  # tuple
        return outputs


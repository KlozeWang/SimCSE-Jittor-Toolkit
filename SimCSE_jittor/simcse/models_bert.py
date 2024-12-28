import jittor
from jittor import nn, init
from .bert_model import BertModel, BertPreTrainedModel

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def execute(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.eps = 1e-8

    def execute(self, x, y):
        x2=(x * x).sum(dim = -1)
        y2=(y * y).sum(dim = -1)
        return (x * y).sum(dim = -1) / jittor.maximum(jittor.sqrt(x2 * y2),self.eps) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def execute(self, attention_mask, outputs):
        last_hidden = outputs["last_hidden_state"]
        pooler_output = outputs["pooler_output"]
        hidden_states = outputs["hidden_states"]

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * jittor.unsqueeze(attention_mask, -1)).sum(1) / jittor.unsqueeze(attention_mask.sum(-1), -1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * jittor.unsqueeze(attention_mask, -1)).sum(1) / jittor.unsqueeze(attention_mask.sum(-1), -1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * jittor.unsqueeze(attention_mask, -1)).sum(1) / jittor.unsqueeze(attention_mask.sum(-1), -1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init()

def sentemb_execute(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return {
        "pooler_output":pooler_output,
        "last_hidden_state":outputs["last_hidden_state"],
        "hidden_states":outputs["hidden_states"],
    }


# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = BertPredictionHeadTransform(config)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         self.bias = init.zero(config.vocab_size)

#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias

#     def execute(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states

def cl_execute(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # Pooling
    
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]
    

    cos_sim = cls.sim(jittor.unsqueeze(z1,1), jittor.unsqueeze(z2,0))

    
    # # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(jittor.unsqueeze(z1,1), jittor.unsqueeze(z3,0))
        cos_sim = jittor.concat([cos_sim, z1_z3_cos], 1)

    labels = jittor.arange(cos_sim.size(0))

    # # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = jittor.array(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        )
        cos_sim = cos_sim + weights

    loss = nn.cross_entropy_loss(cos_sim, labels)
    
    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return {
        "loss":loss,
        "logits":cos_sim,
        "hidden_states":outputs["hidden_states"],
        "attentions":outputs["attentions"],
    }


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, model_kargs):
        super().__init__(config)
        self.model_args = model_kargs
        self.bert = BertModel(config, add_pooling_layer=False)

        # if self.model_args.do_mlm:
        #     self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def execute(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_execute(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_execute(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
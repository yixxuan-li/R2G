import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel
import numpy as np

from .utils import get_siamese_features
from .default_blocks import bert_clf

BERT_PATH = '/home/yixuan/data/bert_pretrain/bert-base-cased'

class Tagger(nn.Module):
    """
        transform the description based on concept vocab to make the description more like concept vocab
    """
    def __init__(self, embedding_size) -> None:
        super(Tagger, self).__init__()

        self.default_embedding = nn.Parameter(torch.rand(embedding_size), requires_grad = True)
        # self.weight = nn.Parameter(torch.eye(embedding_size), requires_grad = True)# embedding_size * embedding_size

    def forward(self, vocab, description):
        """
        args:
            vocab: the concept vocabulary, D x embedding_size,(D: concept number)
            description: the description. B x l x embedding_size, l:description length,
        """
        bts, length, h  = description.shape[:]
        tokens = description #  B x l x embedding_size
        # calculate the similarity between description and concepts 
        # B x l x H, H x H, D+1 x H
        similarity = F.softmax(
            tokens @ (torch.cat([vocab, self.default_embedding.unsqueeze(0)], dim = 0 ).T).unsqueeze(0).repeat(bts, 1, 1),
            dim=2,
        ) #B x l x D+1
        # similarity = F.softmax(
        #     (tokens) @ (torch.cat([vocab, self.default_embedding.unsqueeze(0)], dim = 0 ).T).unsqueeze(0).repeat(bts, 1, 1),
        #     dim=2,
        # ) #B x l x D+1
        # transform the description based on the concept
        #                   B x l x 1        B x l x H    B x l x D             D x H
        concept_based = similarity[:, :, -1:] * tokens + similarity[:, :, :-1] @ vocab # B * l * embedding_size
        return concept_based, similarity[:, :, :]



class InstructionsModel(nn.Module):
    """
    derive the instruction from the description
    """
    def __init__(
        self,
        embedding_size: int,
        n_instructions: int,
        encoded_question_size: int,
        dropout: float = 0.0,
    ) -> None:
        super(InstructionsModel, self).__init__()
        self.embedding_size = embedding_size
        self.tagger = Tagger(embedding_size)
        self.encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=encoded_question_size,
            dropout=dropout,
            batch_first=True
        )
        self.n_instructions = n_instructions
        # self.decoder = nn.RNN(
        #     input_size=encoded_question_size,
        #     hidden_size=embedding_size,
        #     nonlinearity="relu",
        #     dropout=dropout,
        # )
        self.decoder1 = nn.RNNCell(encoded_question_size, embedding_size, bias = False)
        # Use softmax as nn.Module to allow extracting attention weights
        self.softmax = nn.Softmax(dim=-1)
        # self.weight = nn.Parameter(torch.eye(embedding_size), requires_grad = True)


    def forward(self, vocab, description, lang_mask, language_len):
        """
        vocab: C x H, C: sum of conceptions of all properties
        description: B x l x H
        """
        tagged_description, token_similarities = self.tagger(vocab, description) # B * l * embedding_size
        language_embedding = pack_padded_sequence(tagged_description, language_len, batch_first = True)
        _, (encoded, _) = self.encoder(language_embedding)  # get last hidden
        # B x LSTM-encoder-hidden-size
        encoded = encoded.squeeze(dim=0)
        bts, h = encoded.shape[:2]
        # instruction_length x B x LSTM-encoder-hidden-size
        # hidden, _ = self.decoder(encoded.expand(self.n_instructions, -1, -1))
        # hidden = hidden.transpose(0, 1) # B x instruction_length x embedding_size
        inp = encoded
        output = []
        hx = torch.zeros([bts, self.embedding_size], device = 'cuda')
        for _ in range(self.n_instructions):
            hx = self.decoder1(inp, hx)
            output.append(hx.unsqueeze(dim = 1))
        output = torch.cat(output, dim =1)# B * n * H
        # B * n * H @ B * l * H
        lang_mask[lang_mask == 0] = torch.tensor(-np.inf).cuda()
        attention = self.softmax((output @ tagged_description.transpose(1, 2)) + lang_mask.unsqueeze(1))   # B x instruction_length x l
        instructions = attention @ tagged_description   # B x instruction_length x embedding_size
        return instructions, encoded, attention,token_similarities

# class InstructionsModel(nn.Module):
#     """
#     derive the instruction from the description
#     """
#     def __init__(
#         self,
#         embedding_size: int,
#         n_instructions: int,
#         encoded_question_size: int,
#         dropout: float = 0.0,
#     ) -> None:
#         super(InstructionsModel, self).__init__()

#         self.tagger = Tagger(embedding_size)
#         self.encoder = nn.LSTM(
#             input_size=embedding_size,
#             hidden_size=encoded_question_size,
#             dropout=dropout,
#         )
#         self.n_instructions = n_instructions
#         self.decoder = nn.RNN(
#             input_size=encoded_question_size,
#             hidden_size=embedding_size,
#             nonlinearity="relu",
#             dropout=dropout,
#         )
#         # Use softmax as nn.Module to allow extracting attention weights
#         self.softmax = nn.Softmax(dim=-1)


#     def forward(self, vocab, description):
#         """
#         vocab: C x H, C: sum of conceptions of all properties
#         description: B x l x H
#         """
#         tagged_description = self.tagger(vocab, description) # B * l * embedding_size
#         _, (encoded, _) = self.encoder(tagged_description.transpose(0,1))  # get last hidden
#         # B x LSTM-encoder-hidden-size
#         encoded = encoded.squeeze(dim=0)
#         # instruction_length x B x LSTM-encoder-hidden-size
#         hidden, _ = self.decoder(encoded.expand(self.n_instructions, -1, -1))
#         hidden = hidden.transpose(0, 1) # B x instruction_length x embedding_size
#         attention = self.softmax(hidden @ tagged_description.transpose(1, 2))   #B x instruction_length x l
#         instructions = attention @ tagged_description   # B x instruction_length x embedding_size
#         return instructions, encoded

class bert_instruction(nn.Module):
    def __init__(
        self,
        inter_feature_dim = [512, 300]
    )-> None:
        super(bert_instruction, self).__init__()

        inter_layer = []
        pre_dim = 768
        for i, dim in enumerate(inter_feature_dim):
            inter_layer.append(nn.Linear(pre_dim, dim))
            inter_layer.append(nn.Dropout())
            pre_dim = dim
        
        self.ins_layer = nn.Sequential(*inter_layer)

    def forward(
        self,
        text_feature
    ):
        return self.ins_layer(text_feature)
        


class bert_instructions_model(nn.Module):
    def __init__(
        self,
        n_instructions: int,
        inter_feature_dim = None,
        vocab_len = None
    ) -> None:
        super(bert_instructions_model, self).__init__()
        self.embedding_size = 300
        self.n_instructions = n_instructions
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.decoder1 = nn.RNNCell(768, 300, bias = False)
        # Use softmax as nn.Module to allow extracting attention weights
        self.softmax = nn.Softmax(dim=-1)
        self.bert_decoder = bert_clf(300, vocab_len)
        # self.transmation = nn.Parameter(torch.eye(self.embedding_size), requires_grad = True)# embedding_size * embedding_size


    def forward(self, text, attention_mask, vocab):
        outputs = self.bert(text, attention_mask, output_hidden_states=True)
        language_feature = torch.mean(outputs.last_hidden_state, dim = 1).squeeze(1)
        bts, h = language_feature.shape
        instructions = []
        hx = torch.zeros([bts, self.embedding_size], device = 'cuda')
        # for i in range(self.n_ins):
        #     ins = self.instructions_layer[i](outputs.pooler_output)
        #     instructions.append(ins)
        #     ins1 = self.instructions_layer1(language_feature)
        for _ in range(self.n_instructions):
            hx = self.decoder1(language_feature, hx)
            instructions.append(hx.unsqueeze(dim = 1))
        instructions = torch.cat(instructions, dim = 1)
        att_instructions = F.softmax(get_siamese_features(self.bert_decoder, instructions, torch.stack), dim = -1)
        # attention = self.softmax(instructions @ vocab.unsqueeze(0).permute(0,2,1))
        # # attention = self.softmax((instructions @ self.transmation.unsqueeze(0).expand(bts, self.embedding_size, self.embedding_size) ) @ vocab.unsqueeze(0).permute(0,2,1))
        fianl_instruction =  att_instructions @ vocab.unsqueeze(0)

        return outputs.pooler_output, fianl_instruction, att_instructions


class NSMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        dropout: float = 0.0
    ) -> None:
        super(NSMCell, self).__init__()
        self.nonlinearity = nn.ELU()

        # self.weight_node_properties = nn.Parameter(
        #     torch.eye(input_size).unsqueeze(0).repeat(n_node_properties, 1, 1), requires_grad = True
        # )
        self.weight_node_properties = nn.Parameter(
            torch.tensor(np.vstack([np.eye(input_size).reshape(1, input_size, input_size) for i in range(n_node_properties)]) ).to(torch.float32), requires_grad = True
        )
        self.weight_edge = nn.Parameter(torch.eye(input_size), requires_grad = True)
        self.weight_state_score = nn.Parameter(torch.ones(input_size), requires_grad = True)
        self.weight_relation_score = nn.Parameter(torch.ones(input_size), requires_grad = True)
        self.dropout = nn.Dropout(p=dropout)
        # self.weighten_state = nn.Linear(input_size, 1)
        # self.weighten_edge = nn.Linear(input_size, 1)


    def forward(
        self,
        node_attr,
        edge_attr,
        instruction,
        distribution,
        node_prop_similarities,
        relation_prop_similarity,
        node_mask = None
    ):
        """
        Dimensions:
            node_attr: B x N x P x H
            edge_attr: B x N x N x H
            instruction:      B x H
            distribution:          B x N
            node_prop_similarities: B x P(L+1)
            relation_prop_similarity:    B
        Legend:
            N: number of nodes
            P: number of node properties
            H: Hidde/input size (glove embedding size)
            E: number of edges
            B: Batch size
        """
        batch_size, num_node, num_node_properties, dim = node_attr.shape[:4]

        # Compute node and edge score based on the instructions's property relation;
        #  which stand for the node and edge's relative of instruction
        # B x N x H
        node_scores = self.dropout(
            self.nonlinearity(
                torch.sum(
                    # B x P x 1 x 1
                    node_prop_similarities.view(batch_size, -1, 1, 1).expand(batch_size, num_node_properties, num_node, dim)
                    # B x 1 x 1 x H
                    * (instruction.view(batch_size, 1, 1, -1).expand(batch_size, num_node_properties, num_node, dim)
                    # B x P x N x H
                    * (node_attr.transpose(1, 2)
                    # P x H x H -> 1 x P x H x H -> B x P x H x H
                    @ self.weight_node_properties.unsqueeze(0).expand(batch_size, num_node_properties, dim, dim))),
                    dim=1,
                )# B x P x N x H -> B x N x H 
            )
        )

        # E x H
        edge_scores = self.dropout(
            self.nonlinearity(
                (# B x 1 x H
                instruction.view(batch_size, 1, -1).expand(batch_size, num_node*num_node, dim)
                # B x (N x N) x H
                * (edge_attr.view(batch_size, num_node*num_node, -1)
                # H x H -> B x H x H
                @ self.weight_edge.unsqueeze(0).expand(batch_size, dim, dim))).view(batch_size, num_node, num_node, -1)
            )# B x N x N x H
        )
        # shift the attention to their most relavant neibors; B x N x H -> B x N
        # next_distribution_states = F.softmax(self.weighten_state(node_scores).squeeze(2), dim =1)
        next_distribution_states = F.softmax((node_scores @ (self.weight_state_score).view(1, -1, 1).expand(batch_size, dim, 1)).squeeze(2) + node_mask, dim =1)

        # shift the attention to their most relavant edges;  B x N
        # next_distribution_relations = F.softmax(
        #     self.weighten_edge(
        #         torch.sum(
        #             edge_scores * distribution.view(batch_size, 1, -1, 1), dim = 2# (B x N x N x H) * (B x 1 x N x 1)
        #         ).squeeze(2) # B x N x 1 x H -> B x N x H 
        #     ).squeeze(2),# B x N
        #     dim = 1 
        # )
        
        next_distribution_relations = F.softmax(
            (torch.sum(
                    edge_scores * distribution.view(batch_size, 1, -1, 1).expand(batch_size, num_node, num_node, dim), dim = 2# (B x N x N x H) * (B x 1 x N x 1)
                ).squeeze(2)  @ self.weight_relation_score.view(1, -1,1).expand(batch_size, dim, 1)                      # B x N x 1 x H -> B x N x H   @ H
            ).squeeze(2) + node_mask,# B x N
            dim = 1 
        )
        # Compute next distribution
        # B x N
        # next_distribution = F.softmax(
        #  (   relation_prop_similarity.view(batch_size, 1) * next_distribution_relations# (B) x (B X N)
        #     + (1 - relation_prop_similarity).view(batch_size, 1)
        #     * next_distribution_states),  #(B x N)
        #     dim = 1
        # )
        next_distribution = (   relation_prop_similarity.view(batch_size, 1) * next_distribution_relations# (B) x (B X N)
            + (1 - relation_prop_similarity).view(batch_size, 1)
            * next_distribution_states)  #(B x N)
        return next_distribution





## NSM 
class NSM(nn.Module):
    def __init__(self, input_size: int, num_node_properties: int, num_instructions: int, description_hidden_size: int, vocab_len: int, dropout: float = 0.0):
        super(NSM, self).__init__()

        # self.instructions_model = InstructionsModel(
        #     input_size, num_instructions, description_hidden_size, dropout=dropout
        # )#300, 5+1, 16
        self.instructions_model = bert_instructions_model(num_instructions, vocab_len = vocab_len)
        self.nsm_cell = NSMCell(input_size, num_node_properties, dropout=dropout)
        self.W_p = nn.Parameter(torch.eye(input_size), requires_grad = True)
        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        node_attr,
        property_embeddings,# 1 + L +1 
        node_mask,
        description = None,
        edge_attr = None,
        context_size = None,
        lang_mask = None,
        language_len = None,
        concept_vocab_set = None,
        language = None,
        attention_mask = None
    ):
        """
        Dimensions:
            node_attr: B x N x P x H
            edge_attr: B x N x N x H
            description: B x l x H
            concept_vocab: C x H, C: sum of conceptions of all properties
            property_embeddings: D x H
            node_mask: B x N x 1
        Legend:
            B: batch size
            N: Total number of nodes
            P: Number of node properties, L + 1
            H: Hidden size (glove embedding size)
            C: Number of concepts
            D = P + 1: Number of properties (concept categories)
            l: Question length, 
        """
        batch_size, num_node = node_attr.shape[:2]
        num_property = len(property_embeddings)
        ## transform the description to instruction based on concept vocab
        ## instructions: B x instruction_length x embedding_size; encoded_questions:  B x LSTM-encoder-hidden-size
        # instructions, encoded_questions, attention, token_similarities = self.instructions_model(
        #     concept_vocab_set, description, lang_mask, language_len
        # )
        encoded_questions, instructions, attention = self.instructions_model(language.squeeze(1), attention_mask.squeeze(1),concept_vocab_set)


        # Apply dropout to state and edge representations
        # node_attr=self.dropout(node_attr)
        # edge_attr=self.dropout(edge_attr)

        # Initialize distribution over the nodes, size: batch_size x num_node: num of node
        # distribution = F.softmax(torch.rand(batch_size, num_node), dim =1).cuda()
        distribution = torch.ones(batch_size, num_node).cuda() * (1 / context_size).unsqueeze(1)
        # node_mask = 1 - torch.where(torch.isinf(node_mask), torch.full_like(node_mask, 1), node_mask)# B x N
        distribution = F.softmax(distribution + node_mask, dim = -1)
        prob = distribution.unsqueeze(1)
        ins_simi = []
        simi = node_attr[:, :, 0, :].squeeze(2) @ concept_vocab_set.T#B x N x P x H  @ (C x H ).T-> B x N x C
        data, index = torch.sort(simi[:, :, :], dim =-1, descending = True)

        ins_data = []
        ins_index = []

        # # Simulate execution of finite automaton
        for instruction in instructions[:, :].unbind(1):        # B x embedding_size
            # calculate intructions' property similarities(both node and relation)
            # instruction_prop = F.softmax(instruction @ property_embeddings.T, dim=1)  # B x D(L+2)
            instruction_prop = F.softmax(instruction @ (property_embeddings @ self.W_p).transpose(0,1), dim=1)  # B x D(L+2)
            ins_simi.append(instruction_prop.unsqueeze(1))
            # print("---------------")
            # print(instruction_prop[:3])
            # print("---------------")
            node_prop_similarities = instruction_prop[:, :-1]  #B x P(L+1)
            relation_prop_similarity = instruction_prop[:, -1]   # B 
            # distribution = F.softmax(distribution, dim = -1)
            # update the distribution: B xN
            distribution = self.nsm_cell(
                node_attr,
                edge_attr,
                instruction,
                distribution,
                node_prop_similarities,
                relation_prop_similarity,
                node_mask = node_mask
            )
            prob = torch.cat([prob, distribution.unsqueeze(1)], dim =1)

            simi = instruction @ concept_vocab_set.T#B x H  @ (C x H ).T-> B x Cs
            idata, iindex = torch.sort(simi[:, :], dim =-1, descending = True)  
            ins_data.append(idata.unsqueeze(1))
            ins_index.append(iindex.unsqueeze(1))


        ins_data = F.softmax(torch.cat(ins_data, dim = 1), dim =-1)
        ins_index = torch.cat(ins_index, dim = 1)

        # last_instruction = instructions[:, :].unbind(1)[-1]
        ins_simi = torch.cat(ins_simi, dim = 1)
        # print(distribution[:2])
        # arrge = node_attr.view(batch_size, num_node, -1)
    
        """
        # cauculate the final description's node property relation
        # instructions: B x instruction_length x embedding_size; property_embeddings: D x H
        # final_node_prop_similarities = F.softmax(
        #     instructions[:, -1] @ property_embeddings.T, dim=1
        # )[:, :-1]   #B x P;
        
        # update the node feature based on final description and final distribution
        # B x H           # B x N
        # aggregated = distribution.view(batch_size, num_node, 1) * torch.sum(
        #         final_node_prop_similarities.view(batch_size, 1, num_property -1, 1)# B x 1 x P x 1
        #         * node_attr,    #B x N x P x H
        #         dim=2,
        #     ).squeeze(2)# B x N x H
        """
        
        # predictions = self.classifier(torch.hstack((encoded_questions, aggregated)))
        # return torch.cat((extended_encoded_questions, aggregated), dim = -1)
        # final_feature = torch.cat([extended_encoded_questions, arrge ], dim =-1)
        return distribution, encoded_questions, data[:,:, :20], index[:,:, :20], ins_data[:, :, :100], ins_index[:, :, :100], None, attention, ins_simi
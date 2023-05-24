import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from einops import rearrange, repeat

from .utils import get_siamese_features


class Tagger(nn.Module):
    """
        transform the description based on concept vocab to make the description more like concept vocab
    """
    def __init__(self, embedding_size) -> None:
        super(Tagger, self).__init__()

        self.default_embedding = nn.Parameter(torch.rand(embedding_size), requires_grad = True)
        self.weight = nn.Parameter(torch.eye(embedding_size), requires_grad = True)# embedding_size * embedding_size

    def forward(self, vocab, description):
        """
        args:
            vocab: the concept vocabulary, D x embedding_size,(D: concept number)
            description: the description. B x l x embedding_size, l:description length,
        """
        bts, length, h  = description.shape
        tokens = description #  B x l x embedding_size
        # calculate the similarity between description and concepts 
        # B x l x H, H x H, D+1 x H
        similarity = F.softmax(
            torch.matmul(
                        torch.matmul(tokens, repeat(self.weight, 'h1 h2 -> bts h1 h2', bts = bts)),\
                        repeat(torch.cat([vocab, self.default_embedding.unsqueeze(0)], dim = 0 ), 'd h -> b h d', b = bts)
                        ),
            dim=2,
        )
        # transform the description based on the concept
        #                                B x l x 1                                  B x l x H                 B x l x D             D x H
        concept_based = torch.mul(repeat(similarity[:, :, -1], 'b l -> b l h', h =h), tokens) \
                        + torch.matmul(similarity[:, :, :-1],  repeat(vocab, 'd h -> b d h', b = bts)) # B * l * embedding_size
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
        encoded = rearrange(encoded, '1 b h -> b h')
        bts, h = encoded.shape
        # instruction_length x B x LSTM-encoder-hidden-size
        # hidden, _ = self.decoder(encoded.expand(self.n_instructions, -1, -1))
        # hidden = hidden.transpose(0, 1) # B x instruction_length x embedding_size
        inp = encoded
        output = []
        hx = torch.zeros([bts, self.embedding_size], device = 'cuda')
        for _ in range(self.n_instructions):
            hx = self.decoder1(inp, hx)
            output.append(rearrange(hx, 'b h -> b 1 h'))
        output = torch.cat(output, dim =1)# B * n * H
        # B * n * H @ B * l * H
        lang_mask[lang_mask == 0] = torch.tensor(-np.inf).cuda()
        attention = self.softmax(torch.matmul(output, rearrange(tagged_description, 'b l h -> b h l')) \
                                + repeat(lang_mask, 'b l -> b n l', n = self.n_instructions))   # B x instruction_length x l
        instructions = torch.matmul(attention, tagged_description)   # B x instruction_length x embedding_size
        return instructions, encoded, attention, token_similarities

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


class NSMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        dropout: float = 0.0
    ) -> None:
        super(NSMCell, self).__init__()
        self.nonlinearity = nn.ELU()

        self.weight_node_properties = nn.Parameter(
            torch.tensor(np.vstack([np.eye(input_size).reshape(1, input_size, input_size) for i in range(n_node_properties)]) ).to(torch.float32), requires_grad = True
        )
        self.weight_edge = nn.Parameter(torch.eye(input_size), requires_grad = True)
        self.weight_state_score = nn.Parameter(torch.ones(input_size), requires_grad = True)
        self.weight_relation_score = nn.Parameter(torch.ones(input_size), requires_grad = True)
        self.dropout = nn.Dropout(p=dropout)


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
                    torch.mul(repeat(node_prop_similarities, 'b p -> b p n h', n = num_node, h = dim),
                    # B x 1 x 1 x H
                    torch.mul(repeat(instruction, 'b h -> b p n h', p = num_node_properties, n = num_node),
                    # B x P x N x H
                    torch.matmul(rearrange(node_attr, 'b n p h -> b p n h'),
                    # P x H x H -> 1 x P x H x H -> B x P x H x H
                    repeat(self.weight_node_properties, 'p h1 h2 -> b p h1 h2', b = batch_size)))),
                    dim=1,
                )# B x P x N x H -> B x N x H 
            )
        )

        # E x H
        edge_scores = self.dropout(
            self.nonlinearity(
                rearrange(torch.mul(# B x 1 x H
                    repeat(instruction, 'b h -> b nn h', nn = num_node*num_node),
                    # B x (N x N) x H
                    torch.matmul(rearrange(edge_attr, 'b n1 n2 h -> b (n1 n2) h'),
                    # H x H -> B x H x H
                    repeat(self.weight_edge, 'h1 h2 -> b h1 h2', b = batch_size))),
                    'b (n1 n2) h -> b n1 n2 h', n1 = num_node)
            )# B x N x N x H
        )

        # shift the attention to their most relavant neibors; B x N x H -> B x N
        next_distribution_states = F.softmax(rearrange(
                                                        torch.matmul(node_scores, repeat(self.weight_state_score, 'h -> b h 1', b = batch_size)),
                                                        'b n 1 -> b n'
                                                       )
                                             + node_mask, 
                                             dim =1
                                             )

        
        next_distribution_relations = F.softmax(
            rearrange(
                torch.matmul(
                    torch.sum(
                        torch.mul(edge_scores, repeat(distribution, 'b n -> b n1 n h', n1 = num_node, h = dim)), dim = 2),# (B x N x N x H) * (B x 1 x N x 1),
                    repeat(self.weight_relation_score, 'h -> b h 1', b = batch_size)                   # B x N x 1 x H -> B x N x H   @ H
                ),
                'b n 1 -> b n'
                )
            + node_mask,# B x N
            dim = 1 
        )

        next_distribution = ( 
            torch.mul(repeat(relation_prop_similarity, 'b -> b n', n = num_node), next_distribution_relations)# (B) x (B X N)
            + torch.mul(repeat((1 - relation_prop_similarity), 'b -> b n', n = num_node), next_distribution_states))  #(B x N)
        return next_distribution





## NSM 
class NSM(nn.Module):
    def __init__(self,
                 input_size: int, 
                 num_node_properties: int, 
                 num_instructions: int, 
                 description_hidden_size: int, 
                 dropout: float = 0.0,
                 
                 ):
        super(NSM, self).__init__()

        self.instructions_model = InstructionsModel(
            input_size, num_instructions, description_hidden_size, dropout=dropout
        )#300, 5+1, 16
        self.nsm_cell = NSMCell(input_size, num_node_properties, dropout=dropout)
        self.W_p = nn.Parameter(torch.eye(input_size), requires_grad = True)
        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        node_attr,
        description,
        property_embeddings,# 1 + L +1 
        node_mask,
        edge_attr = None,
        context_size = None,
        lang_mask = None,
        language_len = None,
        concept_vocab_set = None
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
        instructions, encoded_questions, attention, token_similarities = self.instructions_model(
            concept_vocab_set, description, lang_mask, language_len
        )

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
        simi = torch.matmul(node_attr[:, :, 0, :], repeat(concept_vocab_set, 'c h -> b h c', b = batch_size))
        data, index = torch.sort(simi[:, :, :], dim =-1, descending = True)

        ins_data = []
        ins_index = []

        # # Simulate execution of finite automaton
        for instruction in instructions[:, :].unbind(1):        # B x embedding_size
            # calculate intructions' property similarities(both node and relation)
            # instruction_prop = F.softmax(torch.matmul(instruction, rearrange(property_embeddings, 'l h -> h l')), dim=1)  # B x D(L+2)
            instruction_prop = F.softmax(torch.matmul(instruction, rearrange(torch.matmul(property_embeddings, self.W_p), 'l h -> h l')), dim=1)  # B x D(L+2)
            ins_simi.append(instruction_prop.unsqueeze(1))
            node_prop_similarities = instruction_prop[:, :-1]  #B x P(L+1)
            relation_prop_similarity = instruction_prop[:, -1]   # B 
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
            #                     B x H                 C x H -> H x C
            simi = torch.matmul(instruction, rearrange(concept_vocab_set, 'c h -> h c'))#B x H  @ (C x H ).T-> B x Cs
            idata, iindex = torch.sort(simi[:, :], dim =-1, descending = True)  
            ins_data.append(idata.unsqueeze(1))
            ins_index.append(iindex.unsqueeze(1))


        ins_data = torch.cat(ins_data, dim = 1)
        ins_index = torch.cat(ins_index, dim = 1)

        last_instruction = instructions[:, :].unbind(1)[-1]
        ins_simi = torch.cat(ins_simi, dim = 1)
        return distribution, encoded_questions, data[:,:, :20], index[:,:, :20], ins_data[:, :, :20], ins_index[:, :, :20], token_similarities, attention, ins_simi
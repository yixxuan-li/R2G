# for load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from .utils import get_siamese_features

class Tagger(nn.Module):
    """
        transform the description based on concept vocab to make the description more like concept vocab
    """
    def __init__(self, embedding_size) -> None:
        super(Tagger, self).__init__()

        self.default_embedding = nn.Parameter(torch.rand(embedding_size))
        self.weight = nn.Parameter(torch.eye(embedding_size))# embedding_size * embedding_size

    def forward(self, vocab, description):
        """
        args:
            vocab: the concept vocabulary, D x embedding_size,(D: concept number)
            description: the description. B x l x embedding_size, l:description length,
        """
        tokens = description #  B x l x embedding_size
        # calculate the similarity between description and concepts 
        # B x l x H, H x H, D+1 x H
        similarity = F.softmax(
            tokens @ self.weight @ torch.cat([vocab, self.default_embedding.unsqueeze(0)], dim = 0 ).T,
            dim=2,
        ) #B x l x D+1
        # transform the description based on the concept
        #                   B x l x 1        B x l x H    B x l x D             D x H
        concept_based = similarity[:, :, -1:] * tokens + similarity[:, :, :-1] @ vocab # B * l * embedding_size
        return concept_based



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

        self.tagger = Tagger(embedding_size)
        self.encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=encoded_question_size,
            dropout=dropout,
        )
        self.n_instructions = n_instructions
        self.decoder = nn.RNN(
            input_size=encoded_question_size,
            hidden_size=embedding_size,
            nonlinearity="relu",
            dropout=dropout,
        )
        # Use softmax as nn.Module to allow extracting attention weights
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, vocab, description):
        """
        vocab: C x H, C: sum of conceptions of all properties
        description: B x l x H
        """
        tagged_description = self.tagger(vocab, description) # B * l * embedding_size
        _, (encoded, _) = self.encoder(tagged_description.transpose(0,1))  # get last hidden
        # B x LSTM-encoder-hidden-size
        encoded = encoded.squeeze(dim=0)
        # instruction_length x B x LSTM-encoder-hidden-size
        hidden, _ = self.decoder(encoded.expand(self.n_instructions, -1, -1))
        hidden = hidden.transpose(0, 1) # B x instruction_length x embedding_size
        attention = self.softmax(hidden @ tagged_description.transpose(1, 2))   #B x instruction_length x l
        instructions = attention @ tagged_description   # B x instruction_length x embedding_size
        return instructions, encoded


class NSMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        dropout: float = 0.0
    ) -> None:
        super(NSMCell, self).__init__()
        self.nonlinearity = nn.Sigmoid()

        self.weight_node_properties = nn.Parameter(
            torch.rand(n_node_properties, input_size, input_size), requires_grad = True
        )
        # self.weight_node_properties2 = nn.Parameter(
        #     torch.rand(n_node_properties, input_size, input_size), requires_grad = True
        # )
        self.weight_edge = nn.Parameter(torch.rand(input_size, input_size), requires_grad = True)
        self.weight_state_score = nn.Parameter(torch.rand(input_size), requires_grad = True)
        # self.weight_state_score2 = nn.Parameter(torch.rand(input_size), requires_grad = True)
        self.weight_relation_score = nn.Parameter(torch.rand(input_size), requires_grad = True)
        self.dropout = nn.Dropout(p=dropout)
        # self.weighten_state = nn.Linear(input_size, 1)
        # self.weighten_edge = nn.Linear(input_size, 1)


    def forward(
        self,
        node_attr,
        edge_attr,
        instruction,
        distribution,
        ins_id,
        node_prop_similarities = None,
        relation_prop_similarity = None,
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
        batch_size, num_node, num_node_properties = node_attr.shape[:3]

        # Compute node and edge score based on the instructions's property relation;
        #  which stand for the node and edge's relative of instruction
        # B x N x H
        if ins_id  == 0:
            # node_scores = self.dropout(
            #     self.nonlinearity(
            #         torch.sum(
            #             # # B x P x 1 x 1
            #             # node_prop_similarities.view(batch_size, -1, 1, 1)
            #             # B x 1 x 1 x H
            #             instruction.view(batch_size, 1, 1, -1)
            #             # B x P x N x H
            #             * node_attr.transpose(1, 2)
            #             # P x H x H
            #             @ self.weight_node_properties,
            #             dim=1,
            #         )# B x P x N x H -> B x N x H 
            #     )
            # )
            tmp = torch.sum(F.normalize(instruction.view(batch_size, 1, 1, -1) * node_attr.transpose(1, 2) @ self.weight_node_properties, dim = 2), dim=1)
            node_scores = self.dropout(self.nonlinearity(tmp))

        if ins_id == 2:
        #     node_scores = self.dropout(
        #         self.nonlinearity(
        #             torch.sum(
        #                 # # B x P x 1 x 1
        #                 # node_prop_similarities.view(batch_size, -1, 1, 1)
        #                 # B x 1 x 1 x H
        #                 instruction.view(batch_size, 1, 1, -1)
        #                 # B x P x N x H
        #                 * node_attr.transpose(1, 2)
        #                 # P x H x H
        #                 @ self.weight_node_properties2,
        #                 dim=1
        #             )# B x P x N x H -> B x N x H 
        #         )
        #     )
            tmp = torch.sum(F.normalize(instruction.view(batch_size, 1, 1, -1) * node_attr.transpose(1, 2) @ self.weight_node_properties, dim = 2), dim=1)
            node_scores = self.dropout(self.nonlinearity(tmp))

        if ins_id %2 != 0:
            # E x H
            # edge_scores = self.dropout(
            #     self.nonlinearity(
            #         (# B x 1 x H
            #         instruction.view(batch_size, 1, -1)
            #         # B x (N x N) x H
            #         * edge_attr.view(batch_size, num_node*num_node, -1)
            #         # H x H
            #         @ self.weight_edge.unsqueeze(0).repeat(batch_size, 1, 1)).view(batch_size, num_node, num_node, -1)
            #     )# B x N x N x H
            # )
            tmp = (F.normalize(instruction.view(batch_size, 1, -1) * edge_attr.view(batch_size, num_node*num_node, -1)@ self.weight_edge.unsqueeze(0).repeat(batch_size, 1, 1), dim = 1)).view(batch_size, num_node, num_node, -1)
            edge_scores = self.dropout(self.nonlinearity(tmp))
            # print(tmp[0,0, :2])
            
        # shift the attention to their most relavant neibors; B x N x H -> B x N
        # next_distribution_states = F.softmax(self.weighten_state(node_scores).squeeze(2), dim =1)
        # if ins_id % 2 == 0:
        #     next_distribution_states = F.softmax((node_scores @ (self.weight_state_score).view(1, -1, 1)).squeeze(2), dim =1)
        if ins_id == 0:
            next_distribution_states = F.softmax((node_scores @ (self.weight_state_score).view(1, -1, 1)).squeeze(2), dim =1)
        if ins_id == 2:
            next_distribution_states = (node_scores @ (self.weight_state_score).view(1, -1, 1)).squeeze(2)
            
            
        # shift the attention to their most relavant edges;  B x N
        # next_distribution_relations = F.softmax(
        #     self.weighten_edge(
        #         torch.sum(
        #             edge_scores * distribution.view(batch_size, 1, -1, 1), dim = 2# (B x N x N x H) * (B x 1 x N x 1)
        #         ).squeeze(2) # B x N x 1 x H -> B x N x H 
        #     ).squeeze(2),# B x N
        #     dim = 1 
        # )
        
        if ins_id %2 != 0:
            next_distribution_relations = F.softmax(
                (torch.sum(
                        edge_scores * distribution.view(batch_size, 1, -1, 1), dim = 2# (B x N x N x H) * (B x 1 x N x 1)
                    ).squeeze(2)  @ self.weight_relation_score.view(1, -1,1)                      # B x N x 1 x H -> B x N x H   @ H
                ).squeeze(2),# B x N
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
        
        # next_distribution = (   relation_prop_similarity.view(batch_size, 1) * next_distribution_relations# (B) x (B X N)
        #     + (1 - relation_prop_similarity).view(batch_size, 1)
        #     * next_distribution_states)  #(B x N)   
        if ins_id %2 == 0:
            next_distribution = next_distribution_states  #(B x N)
            # next_distribution = next_distribution * distribution
        elif ins_id %2 != 0:
            next_distribution = next_distribution_relations#(B X N)
            
        if ins_id == 2:
            next_distribution = next_distribution * distribution
                                          
        return next_distribution






## NSM 
class NSM(nn.Module):
    def __init__(self, input_size: int, 
                 num_node_properties: int, 
                 dropout: float = 0.0,
                 description_hidden_size = None, 
                 num_instructions = None,
                 relation_clf = None,
                 language_clf = None,
                 instruction_clf = None
                 ):
        super(NSM, self).__init__()

        # self.instructions_model = InstructionsModel(
        #     input_size, num_instructions, description_hidden_size, dropout=dropout
        # )#300, 5+1, 16
        self.nsm_cell = NSMCell(input_size, num_node_properties, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # self.anchor_clf = instruction_clf
        # self.lang_relation_classify = relation_clf
        # self.target_clf = language_clf        
        
    def forward(
        self,
        node_attr,
        description,
        concept_vocab,
        concept_vocab_seg,
        property_embeddings,# 1 + L +1 
        node_mask,
        instructions = None,
        edge_attr = None,
        relation_logits = None,
        relation_vocab = None,
        context_size = None
    ):
        """
        Dimensions:
            node_attr: B x N x P x H
            edge_attr: B x N x N x H
            description: B x l x H
            concept_vocab: C x H, C: sum of conceptions of all properties
            property_embeddings: D x H
            node_mask: B x N x 1
            context_size: B x 1
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
        # instructions, encoded_questions = self.instructions_model(
        #     concept_vocab, description
        # )
        
        # ## constrain the 3 instructions
        # anchor_logits = None
        # anchor_instruction = None
        # if self.anchor_clf is not None:
        #     anchor_logits = self.anchor_clf(instructions[:, :].unbind(1)[0])
        #     anchor_instruction = anchor_logits @ concept_vocab[:concept_vocab_seg[0]]

        # lang_relation_logits = None
        # relation_instruction = None
        # if self.lang_relation_classify is not None:
        #     lang_relation_logits = self.lang_relation_classify(instructions[:, :].unbind(1)[1])# B x n_relation
            
        #     # relation_regular = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]).unsqueeze(0).repeat(batch_size, 1).cuda()
            
        #     # relation_regular[:, torch.argmax(lang_relation_logits, dim =-1)] = 0.82
        #     # edge_attr = (relation_logits * relation_regular.view(batch_size,1,1, -1)) @ relation_vocab
            
            
        #     # generate new instruction based on relation predicted
        #     #                   B x n_relations        n_relations x hidden_dim -> B x  hidden_dim
        #     relation_instruction = lang_relation_logits @ concept_vocab[concept_vocab_seg[-2]:]
        #     # instructions[:, 1, :] = new_instruction
            
        # target_logits = None
        # target_instruction = None
        # if self.target_clf is not None:
        #     target_logits = self.target_clf(instructions[:, :].unbind(1)[2])
        #     target_instruction = target_logits @ concept_vocab[:concept_vocab_seg[0]]
            
        

        #  B x LSTM-encoder-hidden-size -> B x num_node x LSTM-encoder-hidden-size
        # extended_encoded_questions = encoded_questions.view(batch_size, 1, -1).repeat(1, num_node, 1)

        # Apply dropout to state and edge representations
        # node_attr=self.dropout(node_attr)
        # edge_attr=self.dropout(edge_attr)

        # Initialize distribution over the nodes, size: batch_size x num_node: num of node
        # distribution = F.softmax(torch.rand(batch_size, num_node), dim =1).cuda()
        
        #                       B x N                           B x 1
        distribution = torch.ones(batch_size, num_node).cuda() * (1 / context_size).unsqueeze(1)
        
        node_mask = 1 - torch.where(torch.isinf(node_mask), torch.full_like(node_mask, 1), node_mask)# B x N
        
        distribution = distribution * node_mask
        
        
        # distribution = F.softmax(get_siamese_features(self.dis, torch.cat([node_attr.view(batch_size, num_node, -1), extended_encoded_questions], dim =-1), torch.stack), dim =-1)
        prob = distribution.unsqueeze(1)
        # # Simulate execution of finite automaton
        # for ins_id, instruction in enumerate(instructions[:, :].unbind(1)):        # B x embedding_size
        for ins_id in range(3):
            # calculate intructions' property similarities(both node and relation)
            # instruction_prop = F.softmax(instruction @ property_embeddings.T, dim=1)  # B x D(L+2)
            # node_prop_similarities = instruction_prop[:, :-1]  #B x P(L+1)
            # relation_prop_similarity = instruction_prop[:, -1]   # B 
            # distribution = F.softmax(distribution, dim = -1)
            # update the distribution: B xN
            instruction = instructions[:, :].unbind(1)[ins_id]
                
                
            distribution = self.nsm_cell(
                node_attr,
                edge_attr,
                instruction,
                distribution,
                ins_id
                # node_prop_similarities,
                # relation_prop_similarity
            )
            prob = torch.cat([prob, distribution.unsqueeze(1)], dim =1)
            
            # if ins_id == 1:
            #     distribution = distribution + t_distribution
            
        all_instruction = instructions[:, :].unbind(1)
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
        return distribution, None, prob, all_instruction, None, None, None
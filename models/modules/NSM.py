# for load
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from einops import rearrange, repeat
from .utils import get_siamese_features
import pdb



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
        _, (encoded, _) = self.encoder(rearrange(tagged_description, 'b l h -> l b h'))  # get last hidden
        # B x LSTM-encoder-hidden-size
        encoded = rearrange(encoded, '1 b h -> b h')
        # instruction_length x B x LSTM-encoder-hidden-size
        hidden, _ = self.decoder(repeat(encoded, 'b h -> n b h', n = self.n_instructions))
        hidden = rearrange(hidden, 'n b h -> b n h') # B x instruction_length x embedding_size
        attention = self.softmax(torch.matmul(hidden, rearrange(tagged_description, 'b l h -> b h l')))   #B x instruction_length x l
        instructions = torch.matmul(attention, tagged_description)    # B x instruction_length x embedding_size
        return instructions, encoded


class NSMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_node_properties: int,
        n_ins: int,
        relation_dim: int,
        dropout: float = 0.0
    ) -> None:
        super(NSMCell, self).__init__()
        self.nonlinearity = nn.ELU()

        self.weight_node_properties = nn.Parameter(
            torch.tensor(np.vstack([np.eye(input_size).reshape(1, input_size, input_size) for i in range(n_node_properties)]) ).to(torch.float32), requires_grad = True
        )
        self.weight_edge = nn.Parameter(torch.eye(relation_dim), requires_grad = True)
        self.weight_state_score = nn.Parameter(torch.ones(input_size), requires_grad = True)
        self.weight_relation_score = nn.Parameter(torch.ones(relation_dim), requires_grad = True)
        self.dropout = nn.Dropout(p=dropout)

        self.n_ins = n_ins
        self.relation_dim = relation_dim


    def forward(
        self,
        node_attr,
        edge_attr,
        instruction,
        distribution,
        ins_id,
        node_prop_similarities = None,
        relation_prop_similarity = None,
        node_mask = None,
        instructions_mask = None
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
        # if (self.n_ins == 3 and ins_id != 1) or (self.n_ins == 21 and ins_id != 10):
        if ins_id != 1:
            node_scores = self.dropout(
                self.nonlinearity(
                    torch.sum(
                            F.normalize(
                                torch.mul(
                                    repeat(node_prop_similarities, 'b p -> b p n h', n = num_node, h = dim),
                                    torch.matmul(
                                        torch.mul(
                                            repeat(instruction, 'b h -> b p n h', p = num_node_properties, n = num_node),
                                            rearrange(node_attr, 'b n p h -> b p n h')
                                            ),
                                        repeat(self.weight_node_properties, 'p h1 h2 -> b p h1 h2', b = batch_size)
                                    )
                                )
                            ),
                        dim=1,
                    )# B x P x N x H -> B x N x H 
                )
            )

        if ins_id == 1:
            # if (self.n_ins == 3 and ins_id == 1) or (self.n_ins == 21 and ins_id == 10):
            edge_scores = self.dropout(
                self.nonlinearity(
                                rearrange(
                                    F.normalize(
                                        torch.matmul(
                                            torch.mul(
                                                repeat(instruction, 'b h -> b nn h', nn = num_node*num_node),
                                                rearrange(edge_attr, 'b n1 n2 h -> b (n1 n2) h')
                                            ),
                                            repeat(self.weight_edge, 'h1 h2 -> b h1 h2', b = batch_size)
                                        )
                                    ), 
                                    'b (n1 n2) h -> b n1 n2 h', n1 = num_node
                                )
                )# B x N x N x H
            )

            
        # shift the attention to their most relavant neibors; B x N x H -> B x N
        # next_distribution_states = F.softmax(self.weighten_state(node_scores).squeeze(2), dim =1)
        # if ins_id % 2 == 0:
        #     next_distribution_states = F.softmax((node_scores @ (self.weight_state_score).view(1, -1, 1)).squeeze(2), dim =1)
        if (self.n_ins == 3 and ins_id == 0):
            next_distribution_states = F.softmax(
                rearrange(
                    torch.matmul(node_scores, repeat(self.weight_state_score, 'h -> b h 1', b = batch_size)), 'b n 1 -> b n'
                ) + node_mask,
                dim = 1
            )
        if (self.n_ins == 3 and ins_id == 2) or (self.n_ins == 19 and ins_id != 9):
            next_distribution_states = rearrange(
                    torch.matmul(node_scores, repeat(self.weight_state_score, 'h -> b h 1', b = batch_size)), 'b n 1 -> b n'
                )
    
            
        # shift the attention to their most relavant edges;  B x N
        # next_distribution_relations = F.softmax(
        #     self.weighten_edge(
        #         torch.sum(
        #             edge_scores * distribution.view(batch_size, 1, -1, 1), dim = 2# (B x N x N x H) * (B x 1 x N x 1)
        #         ).squeeze(2) # B x N x 1 x H -> B x N x H 
        #     ).squeeze(2),# B x N
        #     dim = 1 
        # )
        
        if (self.n_ins == 3 and ins_id == 1) or (self.n_ins == 19 and ins_id == 9):
            next_distribution_relations = F.softmax(
                (torch.sum(
                        edge_scores * distribution.view(batch_size, 1, -1, 1).expand(batch_size, num_node, num_node, self.relation_dim), dim = 2# (B x N x N x H) * (B x 1 x N x 1)
                    ).squeeze(2)  @ self.weight_relation_score.view(1, -1,1).expand(batch_size, self.relation_dim, 1)                      # B x N x 1 x H -> B x N x H   @ H
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
        
        # next_distribution = (   relation_prop_similarity.view(batch_size, 1) * next_distribution_relations# (B) x (B X N)
        #     + (1 - relation_prop_similarity).view(batch_size, 1)
        #     * next_distribution_states)  #(B x N)   

        if (self.n_ins == 3 and ins_id != 1) or (self.n_ins == 19 and ins_id != 9):
            next_distribution = next_distribution_states  #(B x N)
        elif (self.n_ins == 3 and ins_id == 1) or (self.n_ins == 19 and ins_id == 9):
            if instructions_mask is not None:
                next_distribution = torch.mul(next_distribution_relations, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node))#(B X N)
            else:
                next_distribution = next_distribution_relations
                
        if (self.n_ins == 3 and ins_id == 2):
            next_distribution = torch.mul(next_distribution, distribution)

        # if instructions_mask is not None:
        #     if (self.n_ins == 19 and ins_id == 8):
        #         next_distribution = F.softmax(torch.mul(next_distribution, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node)) + distribution, dim = -1)
        #     elif (self.n_ins == 19 and ins_id!= 9):
        #         next_distribution = torch.mul(next_distribution, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node)) + distribution
        # else:
        #     if (self.n_ins == 19 and ins_id == 8):
        #         next_distribution = next_distribution + distribution
        #     elif (self.n_ins == 19 and ins_id!= 9):
        #         next_distribution = next_distribution + distribution




        if instructions_mask is not None:
            if (self.n_ins == 19 and ins_id == 8):
                next_distribution = F.softmax(torch.mul(next_distribution, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node)) + distribution, dim = -1)
            # elif (self.n_ins == 19 and ins_id == 10):
            #     next_distribution = torch.mul(next_distribution, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node)) + distribution
            elif (self.n_ins == 19 and ins_id!= 9):
                next_distribution = F.softmax(torch.mul(next_distribution, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node)) + distribution, dim = -1)
        else:
            if (self.n_ins == 19 and ins_id == 8):
                next_distribution = F.softmax(next_distribution + distribution, dim =-1)
            elif (self.n_ins == 19 and ins_id == 9):
                next_distribution = F.softmax(torch.mul(next_distribution, distribution), dim =-1)
            elif (self.n_ins == 19 and ins_id!= 9):
                next_distribution = next_distribution + distribution


            # next_distribution = next_distribution + distribution   
        # next_distribution = torch.mul(next_distribution, repeat(instructions_mask[:,ins_id], 'b -> b n', n = num_node))
        # if (self.n_ins == 19 and ins_id == 8):
        #     next_distribution = F.softmax(next_distribution, dim = -1)
        # print(next_distribution[0, :])
        return next_distribution






## NSM 
class NSM(nn.Module):
    def __init__(self, 
                 args,
                 input_size: int, 
                 num_node_properties: int, 
                 num_instructions: int, 
                 description_hidden_size: int, 
                 dropout: float = 0.0,
                 relation_dim = int,
                 vocab_len = int,
                 anchor_clf = None,
                 relation_clf = None,
                 target_clf = None
                 ):
        super(NSM, self).__init__()
        if not args.use_LLM:
            self.instructions_model = InstructionsModel(
                input_size, num_instructions, description_hidden_size, dropout=dropout
            )#300, 5+1, 16
            self.anchor_clf = anchor_clf
            self.relation_clf = relation_clf
            self.target_clf = target_clf    

        self.num_instructions = num_instructions
        self.nsm_cell = NSMCell(input_size, num_node_properties, n_ins = num_instructions, relation_dim = relation_dim, dropout=dropout)  
        self.dropout = nn.Dropout(dropout)  
        
    def forward(
        self,
        args,
        node_attr,
        description,
        concept_vocab,
        concept_vocab_seg,
        property_embeddings,# 1 + L +1 
        node_mask,
        edge_attr = None,
        relation_logits = None,
        relation_vocab = None,
        context_size = None,
        lang_mask = None,
        language_len = None,
        concept_vocab_set = None,
        instructions = None,
        instructions_mask = None,
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


        anchor_logits = None
        anchor_instruction = None
        lang_relation_logits = None
        relation_instruction = None
        target_logits = None
        target_instruction = None
        encoded_questions = None
        if not args.use_LLM:
            ## transform the description to instruction based on concept vocab
            ## instructions: B x instruction_length x embedding_size; encoded_questions:  B x LSTM-encoder-hidden-size
            instructions, encoded_questions = self.instructions_model(
                concept_vocab, description
            )
    
            ## constrain the 3 instructions
            if self.anchor_clf is not None:
                anchor_logits = self.anchor_clf(instructions[:, :].unbind(1)[0])
                anchor_instruction = torch.matmul(anchor_logits, concept_vocab[:concept_vocab_seg[0]])

            if self.relation_clf is not None:
                lang_relation_logits = self.relation_clf(instructions[:, :].unbind(1)[1])# B x n_relation
                relation_instruction = torch.matmul(lang_relation_logits, relation_vocab)
                
            if self.target_clf is not None:
                if self.num_instructions == 3:
                    target_logits = self.target_clf(instructions[:, :].unbind(1)[2])
                    target_instruction = torch.matmul(target_logits, concept_vocab[:concept_vocab_seg[0]])
                elif self.num_instructions == 19:
                    target_logits = self.target_clf(instructions[:, :].unbind(1)[10])
                    target_instruction = torch.matmul(target_logits, concept_vocab[:concept_vocab_seg[0]])

        # Apply dropout to state and edge representations
        # node_attr=self.dropout(node_attr)
        # edge_attr=self.dropout(edge_attr)

        # Initialize distribution over the nodes, size: batch_size x num_node: num of node
        
        #                       B x N                           B x 1
        distribution = torch.ones(batch_size, num_node).cuda() * (1 / context_size).unsqueeze(1)
        
        distribution = F.softmax(distribution + node_mask, dim = -1)
        
        
        prob = distribution.unsqueeze(1)
        # # Simulate execution of finite automation
        # for ins_id, instruction in enumerate(instructions[:, :].unbind(1)):        # B x embedding_size
        for ins_id in range(self.num_instructions):
            # calculate intructions' property similarities(both node and relation)
            if args.implicity_instruction:
                instruction = instructions[:, :].unbind(1)[ins_id]
                instruction_prop = F.softmax(instruction @ property_embeddings.T, dim=1)  # B x D(L+2)
            else:
                instruction_prop = torch.zeros([batch_size, num_property]).cuda()
                if self.num_instructions == 3:
                    if ins_id == 1:
                        instruction_prop[:, -1] = 1
                    else:
                        instruction_prop[:, :-1] = 1
                elif self.num_instructions == 19:
                    if ins_id == 9:
                        instruction_prop[:, -1] = 1
                    else:
                        instruction_prop[:, ins_id % 10] = 1
            # if instructions_mask is not None:
            #     instruction_prop = torch.mul(instruction_prop,  repeat(instructions_mask[:, ins_id], 'b -> b n', n = num_property))
            node_prop_similarities = instruction_prop[:, :-1]  #B x P(L+1)
            relation_prop_similarity = instruction_prop[:, -1]   # B 
            # update the distribution: B xN
            instruction = instructions[:, :].unbind(1)[ins_id]

            if ins_id == 0 and anchor_instruction is not None:
                instruction = anchor_instruction
            if ins_id == 1 and relation_instruction is not None:
                instruction = relation_instruction
            if ins_id == 2 and target_instruction is not None:
                instruction = target_instruction
                
            distribution = self.nsm_cell(
                node_attr,
                edge_attr,
                instruction,
                distribution,
                ins_id,
                node_mask = node_mask,
                node_prop_similarities = node_prop_similarities,
                relation_prop_similarity = relation_prop_similarity,
                instructions_mask = instructions_mask
            )
            prob = torch.cat([prob, rearrange(distribution, 'b n -> b 1 n')], dim =1)
            
            
        all_instruction = instructions[:, :].unbind(1)
        # print("**********")

        return distribution, encoded_questions, prob, all_instruction, anchor_logits, lang_relation_logits, target_logits

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 15:26:27 2025

@author: korotkevich
"""
import model
import torch
from segments import SimpleNetwork, WeightedPooling
import torch.nn as nn
from core import BaseModelClass

class ReacRoost(model.Roost):
    
    def __init__(self,
            task,
            robust,
            n_targets,
            elem_emb_len,
            elem_fea_len=64,
            n_graph=3,
            elem_heads=3,
            elem_gate=[256],
            elem_msg=[256],
            cry_heads=3,
            cry_gate=[256],
            cry_msg=[256],
            out_hidden=[1024, 512, 256, 128, 64],
            dim_red = True,
            append_after = "C", #C OR E
            append_how_many = 2, #2 loadings
            **kwargs
        ):
        
        super().__init__(task=task, 
                       robust=robust, 
                       n_targets=n_targets, 
                       elem_emb_len = elem_emb_len,
                       elem_fea_len=elem_fea_len,
                       n_graph = n_graph,
                       elem_heads = elem_heads,
                       elem_gate = elem_gate,
                       elem_msg = elem_msg,
                       cry_heads = cry_heads,
                       cry_gate = cry_gate,
                       cry_msg = cry_msg,
                       out_hidden = out_hidden,
                       **kwargs)
        
        
        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
            "dim_red": dim_red,
            "append_after" : append_after, #C OR E
            "append_how_many" : append_how_many, #2 loadings
            }
        
        self.append_after = append_after
        
        self.model_params.update(desc_dict)
        
        
        self.material_nn = ReacDescriptorNetwork(**desc_dict)
        
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets
        
        self.dim_red = dim_red
        
        #Dimension of the output network dependent on whether it's ReacRoostC or ReacRoostE
        if append_after == "C":
            if self.dim_red:
                self.output_nn = SimpleNetwork(elem_fea_len + append_how_many, output_dim, out_hidden, nn.ReLU)
            else:
                self.output_nn = SimpleNetwork(elem_fea_len + append_how_many + 1, output_dim, out_hidden, nn.ReLU)
        elif append_after == "E":
            if self.dim_red:
                self.output_nn = SimpleNetwork(elem_fea_len, output_dim, out_hidden, nn.ReLU)
            else:
                self.output_nn = SimpleNetwork(elem_fea_len + append_how_many + 1, output_dim, out_hidden, nn.ReLU)
        
    def forward(self,
                elem_weights,
                elem_fea,
                self_fea_idx,
                nbr_fea_idx,
                cry_elem_idx,
                appended_vals):

        """
        Forward pass through the material_nn and output_nn
        """
        crys_fea = self.material_nn(
            elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx, appended_vals
        )

        #Append hydration numbers to the material descriptor in case it's ReacRoostC
        if self.append_after == "C":
            crys_fea = torch.cat((crys_fea, appended_vals.float()), dim = 1)
            
        elif self.append_after == "E":
            pass

        return self.output_nn(crys_fea)
    
class ReacDescriptorNetwork(model.DescriptorNetwork):
    
    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        dim_red = True,
        append_after = "C",
        append_how_many = 2
    ):
        """
        """
        self.dim_red = dim_red
        self.append_after = append_after
        
        if not self.dim_red:
            #Specify dimension for the MP layers after appending the atomic fraction
            #and the hydration states in case of ReacRoostE
            if self.append_after == "C":
                elem_fea_len = elem_fea_len + 1
            elif self.append_after == "E":
                elem_fea_len = elem_fea_len + 1 + append_how_many
            
        #Initialize MP layers with the correct feature length
        super().__init__(elem_emb_len,
                elem_fea_len,
                n_graph,
                elem_heads,
                elem_gate,
                elem_msg,
                cry_heads,
                cry_gate,
                cry_msg)
        
        #Reinstate the dimred step
        if self.dim_red:
            #Specify dimred transformation, keeping the dimension within the MP
            #layers similar for ReacRoostC and ReacRoostE
            if append_after == "C":
                self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)
            elif append_after == "E":
                self.embedding = nn.Linear(elem_emb_len, elem_fea_len - append_how_many - 1)
        
        
                
    def forward(self, 
                elem_weights,
                elem_fea,
                self_fea_idx,
                nbr_fea_idx,
                cry_elem_idx,
                appended_vals):
        
        #Trainable dim. reduction step of the original features 
        if self.dim_red:
            elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # add hydration states if it's ReacRoostE
        if self.append_after == "C":
            pass
        elif self.append_after == "E":

            elem_fea = torch.cat([elem_fea, appended_vals.float()], dim=1)

        
        #Apply the message passing functions (as in the original Roost)
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)

            )

        return torch.mean(torch.stack(head_fea), dim=0)
    
class ReacElemNet(BaseModelClass):
    
    def __init__(
        self,
        task,
        robust,
        n_targets,
        elem_emb_len,
        elem_fea_len=64,
        out_hidden=[1024, 512, 256, 128, 64],
        dim_red = True,
        append_how_many = 2, #hydration states
        **kwargs
    ):
        super().__init__(task=task, robust=robust, n_targets=n_targets, **kwargs)
        
        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len
        }
        

        self.material_nn = ReacDescriptor(**desc_dict)

        
        self.model_params.update(
            {
                "task": task,
                "robust": robust,
                "n_targets": n_targets,
                "out_hidden": out_hidden
            }
        )
        
        self.model_params.update(desc_dict)
        
        # define an output neural network
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets

        self.output_nn = SimpleNetwork(elem_fea_len + append_how_many, output_dim, out_hidden, nn.ReLU)
        
    def forward(self,
                elem_weights,
                elem_fea,
                self_fea_idx,
                nbr_fea_idx,
                cry_elem_idx,
                appended_vals):
        

        crys_fea = self.material_nn(elem_weights, elem_fea, cry_elem_idx)

        #Append hydration numbers to the material descriptor
        crys_fea = torch.cat((crys_fea, appended_vals.float()), dim = 1)
        
        #apply neural network to map from learned features to target
        return self.output_nn(crys_fea)
    
    def __repr__(self):
        return self.__class__.__name__
    
class ReacDescriptor(nn.Module):
    
    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        dim_red = True
    ):
        """
        """
        super().__init__()
        
        self.dim_red = dim_red
        
        if self.dim_red:
            self.embedding = nn.Linear(elem_emb_len, elem_fea_len)
            
        self.cry_pool = WeightedPooling()
                

    def forward(self, elem_weights, elem_fea, cry_elem_idx):
           

        # embed the original features into a trainable embedding space
        # dim reduction step
        if self.dim_red:
            elem_fea = self.embedding(elem_fea)

        #add weights as a node feature
        # elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # generate crystal features by pooling the elemental features
        crys_fea = self.cry_pool(elem_fea, index=cry_elem_idx, weights=elem_weights)

        
        return crys_fea
    
    def __repr__(self):
        return self.__class__.__name__

class ReacCryNet(BaseModelClass):
    
    def __init__(self,
            task,
            robust,
            n_targets,
            cry_emb_len,
            cry_fea_len=64,
            out_hidden=[1024, 512, 256, 128, 64],
            dim_red = True,
            append_how_many = 2, #hydration states
            **kwargs):
        
        super().__init__(task=task, robust=robust, n_targets=n_targets, **kwargs)
        
        desc_dict = {
            "cry_emb_len": cry_emb_len,
            "cry_fea_len": cry_fea_len,
            "dim_red": dim_red
        }
        
        
        self.model_params.update(
            {
                "task": task,
                "robust": robust,
                "n_targets": n_targets,
                "out_hidden": out_hidden
            }
        )
        
        self.model_params.update(desc_dict)
        
        # define an output neural network
        if self.robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets
        
        self.dim_red = dim_red
        
        if self.dim_red:
            self.embedding = nn.Linear(cry_emb_len, cry_fea_len)
        
        self.output_nn = SimpleNetwork(cry_fea_len + append_how_many, output_dim, out_hidden, nn.ReLU)
        
    def forward(self,
                cry_fea,
                appended_vals):
            
            if self.dim_red:
                cry_fea = self.embedding(cry_fea)
            
            #Append hydration numbers to the material descriptor
            cry_fea = torch.cat((cry_fea, appended_vals.float()), dim = 1)
            
            #apply neural network to map from learned features to target
            return self.output_nn(cry_fea)
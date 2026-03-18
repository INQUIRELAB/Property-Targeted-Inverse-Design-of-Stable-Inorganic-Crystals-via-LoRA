# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.types import PropertySourceId
from mattergen.denoiser import GemNetTDenoiser, get_chemgraph_from_denoiser_output
from mattergen.property_embeddings import (
    ZerosEmbedding,
    get_property_embeddings,
    get_use_unconditional_embedding,
)
from mattergen.common.gemnet.gemnet_lora_ctrl import GemNetTLoRACtrl

BatchTransform = Callable[[ChemGraph], ChemGraph]


class GemNetTLoRAAdapter(GemNetTDenoiser):
    """
    LoRA-based denoiser adapter with GemNetT. On top of a mattergen.denoiser.GemNetTDenoiser,
    additionally inputs <property_embeddings_adapt> that specifies extra conditions to be conditioned on
    using LoRA (Low-Rank Adaptation) instead of FiLM adapters.

    Key improvements over FiLM:
    1. Parameter efficiency: LoRA uses much fewer parameters
    2. Better generalization: Low-rank adaptation often generalizes better
    3. Easier fine-tuning: LoRA is more stable for fine-tuning
    4. Modularity: Can easily add/remove adapters for different properties
    """

    def __init__(
        self, 
        property_embeddings_adapt: torch.nn.ModuleDict, 
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        property_dims: dict = None,
        gemnet: torch.nn.Module = None,
        *args, 
        **kwargs
    ):
        # Import AtomEmbedding
        from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
        
        # Extract gemnet kwargs before calling super
        gemnet_kwargs = {}
        if gemnet is not None:
            # If gemnet is provided, extract its configuration
            if hasattr(gemnet, 'state_dict'):
                # This is an instantiated module, we need to create a new LoRA version
                gemnet_kwargs = {
                    'atom_embedding': AtomEmbedding(emb_size=512, with_mask_type=True),
                    'emb_size_atom': getattr(gemnet, 'emb_size_atom', 512),
                    'emb_size_edge': getattr(gemnet, 'emb_size_edge', 512),
                    'latent_dim': getattr(gemnet, 'latent_dim', 512),
                    'num_blocks': getattr(gemnet, 'num_blocks', 4),
                    'cutoff': getattr(gemnet, 'cutoff', 7.0),
                    'max_neighbors': getattr(gemnet, 'max_neighbors', 50),
                    'num_targets': getattr(gemnet, 'num_targets', 1),
                    'regress_stress': getattr(gemnet, 'regress_stress', True),
                    'otf_graph': getattr(gemnet, 'otf_graph', True),
                }
        else:
            # Use default GemNetT parameters
            gemnet_kwargs = {
                'atom_embedding': AtomEmbedding(emb_size=512, with_mask_type=True),
                'emb_size_atom': 512,
                'emb_size_edge': 512,
                'latent_dim': 512,
                'num_blocks': 4,
                'cutoff': 7.0,
                'max_neighbors': 50,
                'num_targets': 1,
                'regress_stress': True,
                'otf_graph': True,
            }
        
        # Create LoRA-enabled GemNet
        lora_gemnet = GemNetTLoRACtrl(
            condition_on_adapt=list(property_embeddings_adapt.keys()),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            property_dims=property_dims,
            **gemnet_kwargs
        )
        
        # Initialize parent with the LoRA-enabled gemnet
        super().__init__(gemnet=lora_gemnet, *args, **kwargs)

        # ModuleDict[PropertyName, PropertyEmbedding] -- conditions adding by this adapter
        self.property_embeddings_adapt = torch.nn.ModuleDict(property_embeddings_adapt)

        # sanity check keys are required by the adapter that already exist in the base model
        assert all(
            [
                k not in self.property_embeddings.keys()
                for k in self.property_embeddings_adapt.keys()
            ]
        ), f"One of adapter conditions {self.property_embeddings_adapt.keys()} already exists in base model {self.property_embeddings.keys()}, please remove."

        # we make the choice that new adapter fields do not alter the unconditional score
        # we therefore need the unconditional embedding for all properties added in the adapter
        # to return 0. We hack the unconditional embedding module here to achieve that
        for property_embedding in self.property_embeddings_adapt.values():
            property_embedding.unconditional_embedding_module = ZerosEmbedding(
                hidden_dim=property_embedding.unconditional_embedding_module.hidden_dim,
            )

    def forward(
        self,
        x: ChemGraph,
        t: torch.Tensor,
    ) -> ChemGraph:
        """
        Forward pass with LoRA adapters.
        """
        (frac_coords, lattice, atom_types, num_atoms, batch,) = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        # (num_atoms, hidden_dim) (num_cryst, 3)
        t_enc = self.noise_level_encoding(t).to(lattice.device)
        z_per_crystal = t_enc

        # shape = (Nbatch, sum(hidden_dim of all properties in condition_on_adapt))
        conditions_base_model: torch.Tensor = get_property_embeddings(
            property_embeddings=self.property_embeddings, batch=x
        )

        if len(conditions_base_model) > 0:
            z_per_crystal = torch.cat([z_per_crystal, conditions_base_model], dim=-1)

        # compose into a dict
        conditions_adapt_dict = {}
        conditions_adapt_mask_dict = {}
        for cond_field, property_embedding in self.property_embeddings_adapt.items():
            conditions_adapt_dict[cond_field] = property_embedding.forward(batch=x)
            try:
                conditions_adapt_mask_dict[cond_field] = get_use_unconditional_embedding(
                    batch=x, cond_field=cond_field
                )
            except KeyError:
                # no values have been provided for the conditional field,
                # interpret this as the user wanting an unconditional score
                conditions_adapt_mask_dict[cond_field] = torch.ones_like(
                    x["num_atoms"], dtype=torch.bool
                ).reshape(-1, 1)

        output = self.gemnet(
            z=z_per_crystal,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch,
            lengths=None,
            angles=None,
            lattice=lattice,
            # we construct the graph on the fly, hence pass None for these:
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            cond_adapt=conditions_adapt_dict,
            cond_adapt_mask=conditions_adapt_mask_dict,  # when True use unconditional embedding
        )

        pred_atom_types = self.fc_atom(output.node_embeddings)

        return get_chemgraph_from_denoiser_output(
            pred_atom_types=pred_atom_types,
            pred_lattice_eps=output.stress,
            pred_cart_pos_eps=output.forces,
            training=self.training,
            element_mask_func=self.element_mask_func,
            x_input=x,
        )

    @property
    def cond_fields_model_was_trained_on(self) -> list[PropertySourceId]:
        """
        We adopt the convention that all property embeddings are stored in torch.nn.ModuleDicts of
        name property_embeddings or property_embeddings_adapt in the case of a fine tuned model.

        This function returns the list of all field names that a given score model was trained to
        condition on.
        """
        return list(self.property_embeddings) + list(self.property_embeddings_adapt)

    def get_lora_parameters(self):
        """Get only the LoRA parameters for optimization."""
        return self.gemnet.get_lora_parameters()

    def freeze_base_model(self):
        """Freeze the base GemNet model, keeping only LoRA parameters trainable."""
        self.gemnet.freeze_base_model()

    def merge_lora_weights(self):
        """Merge LoRA weights into the base model for inference."""
        self.gemnet.merge_lora_weights()

    def get_adaptation_parameter_count(self) -> int:
        """Get the number of trainable parameters in LoRA adapters."""
        return self.gemnet.get_adaptation_parameter_count()

    def get_base_parameter_count(self) -> int:
        """Get the number of parameters in the base model."""
        return self.gemnet.get_base_parameter_count()

    def get_parameter_efficiency_ratio(self) -> float:
        """Get the ratio of LoRA parameters to base model parameters."""
        return self.gemnet.get_parameter_efficiency_ratio()

    def print_parameter_efficiency(self):
        """Print parameter efficiency statistics."""
        lora_params = self.get_adaptation_parameter_count()
        base_params = self.get_base_parameter_count()
        efficiency_ratio = self.get_parameter_efficiency_ratio()
        
        print(f"LoRA Adapter Parameter Statistics:")
        print(f"  Base model parameters: {base_params:,}")
        print(f"  LoRA adapter parameters: {lora_params:,}")
        print(f"  Parameter efficiency ratio: {efficiency_ratio:.4f} ({efficiency_ratio*100:.2f}%)")
        print(f"  Memory reduction: {(1-efficiency_ratio)*100:.2f}%")


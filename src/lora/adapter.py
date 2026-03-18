# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer implementation.
    
    LoRA decomposes weight updates as: ΔW = BA, where B ∈ R^(d×r) and A ∈ R^(r×k)
    with r << min(d,k) being the rank.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = False,
        init_weights: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
            
        if init_weights:
            self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights."""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_A.weight, std=1.0)
        # Initialize B with zeros so initial adaptation is zero
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        # LoRA computation: x @ A^T @ B^T
        result = self.lora_B(self.lora_A(self.dropout(x)))
        result = result * self.scaling
        
        if self.bias is not None:
            result = result + self.bias
            
        return result


class LoRAAdapter(nn.Module):
    """
    LoRA adapter for GemNet layers that can be applied to any linear layer.
    """
    
    def __init__(
        self,
        target_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        adapter_name: str = "lora_adapter"
    ):
        super().__init__()
        
        self.target_layer = target_layer
        self.adapter_name = adapter_name
        self.rank = rank
        self.alpha = alpha
        
        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=target_layer.in_features,
            out_features=target_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=target_layer.bias is not None
        )
        
        # Freeze original layer
        for param in target_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA adaptation."""
        # Original forward pass
        original_output = self.target_layer(x)
        
        # LoRA adaptation
        lora_output = self.lora(x)
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into the original layer (for inference)."""
        with torch.no_grad():
            # Get LoRA weights
            lora_A = self.lora.lora_A.weight  # [rank, in_features]
            lora_B = self.lora.lora_B.weight  # [out_features, rank]
            
            # Compute adaptation: ΔW = B @ A
            adaptation = lora_B @ lora_A  # [out_features, in_features]
            adaptation = adaptation * self.lora.scaling
            
            # Add to original weights
            self.target_layer.weight.data += adaptation
            
            # Handle bias if present
            if self.lora.bias is not None and self.target_layer.bias is not None:
                self.target_layer.bias.data += self.lora.bias.data


class PropertyLoRAAdapter(nn.Module):
    """
    Property-specific LoRA adapter that conditions on property embeddings.
    Similar to FiLM but using LoRA instead of full linear layers.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        property_dim: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.property_dim = property_dim
        self.rank = rank
        
        # Property embedding projection
        self.property_proj = nn.Linear(property_dim, hidden_dim)
        
        # LoRA for feature modulation
        self.lora_modulation = LoRALayer(
            in_features=hidden_dim * 2,  # [hidden, property_projected]
            out_features=hidden_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Gating mechanism (similar to FiLM)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        property_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply LoRA-based property conditioning.
        
        Args:
            features: [N, hidden_dim] - input features
            property_embedding: [N, property_dim] - property conditioning
            mask: [N, 1] - mask for conditional vs unconditional (1=conditional, 0=unconditional)
        """
        # Project property embedding to hidden dimension
        prop_proj = self.property_proj(property_embedding)  # [N, hidden_dim]
        
        # Concatenate features and property projection
        combined = torch.cat([features, prop_proj], dim=-1)  # [N, 2*hidden_dim]
        
        # Apply LoRA modulation
        modulation = self.lora_modulation(combined)  # [N, hidden_dim]
        
        # Apply gating
        gate = self.gate(combined)  # [N, hidden_dim]
        
        # Apply modulation with gating
        adapted_features = features + gate * modulation
        
        # Apply mask if provided
        if mask is not None:
            # mask: 1 = use conditional, 0 = use unconditional
            adapted_features = mask * adapted_features + (1 - mask) * features
        
        return adapted_features


class GemNetLoRAAdapter(nn.Module):
    """
    LoRA adapter system for GemNet that replaces FiLM adapters.
    """
    
    def __init__(
        self,
        condition_on_adapt: List[str],
        hidden_dim: int,
        property_dims: Dict[str, int],
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.condition_on_adapt = condition_on_adapt
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Create LoRA adapters for each property and each block
        self.property_adapters = nn.ModuleDict()
        
        for prop_name in condition_on_adapt:
            prop_dim = property_dims.get(prop_name, hidden_dim)
            
            # Create adapters for each block
            block_adapters = nn.ModuleList()
            for _ in range(num_blocks):
                adapter = PropertyLoRAAdapter(
                    hidden_dim=hidden_dim,
                    property_dim=prop_dim,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                block_adapters.append(adapter)
            
            self.property_adapters[prop_name] = block_adapters
    
    def forward(
        self,
        features: torch.Tensor,
        cond_adapt: Optional[Dict[str, torch.Tensor]] = None,
        cond_adapt_mask: Optional[Dict[str, torch.Tensor]] = None,
        block_idx: int = 0
    ) -> torch.Tensor:
        """
        Apply LoRA adaptation for a specific block.
        
        Args:
            features: [N, hidden_dim] - input features
            cond_adapt: Dict of property embeddings
            cond_adapt_mask: Dict of masks for each property
            block_idx: Which block we're adapting
        """
        if cond_adapt is None or cond_adapt_mask is None:
            return features
        
        adapted_features = features.clone()
        
        for prop_name in self.condition_on_adapt:
            if prop_name in cond_adapt and prop_name in cond_adapt_mask:
                # Get the adapter for this property and block
                adapter = self.property_adapters[prop_name][block_idx]
                
                # Apply LoRA adaptation
                prop_adapted = adapter(
                    features=adapted_features,
                    property_embedding=cond_adapt[prop_name],
                    mask=cond_adapt_mask[prop_name]
                )
                
                # Add to accumulated features
                adapted_features = adapted_features + prop_adapted
        
        return adapted_features
    
    def get_adaptation_parameters(self) -> List[torch.nn.Parameter]:
        """Get only the LoRA parameters for optimization."""
        params = []
        for prop_adapters in self.property_adapters.values():
            for adapter in prop_adapters:
                params.extend(adapter.parameters())
        return params
    
    def merge_adapters(self):
        """Merge all LoRA weights for inference."""
        for prop_adapters in self.property_adapters.values():
            for adapter in prop_adapters:
                if hasattr(adapter.lora_modulation, 'merge_weights'):
                    adapter.lora_modulation.merge_weights()


def create_lora_adapters_for_gemnet(
    gemnet_model: nn.Module,
    condition_on_adapt: List[str],
    property_dims: Dict[str, int],
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None
) -> Dict[str, LoRAAdapter]:
    """
    Create LoRA adapters for specific modules in a GemNet model.
    
    Args:
        gemnet_model: The GemNet model to adapt
        condition_on_adapt: List of property names to condition on
        property_dims: Dictionary mapping property names to their embedding dimensions
        rank: LoRA rank
        alpha: LoRA alpha scaling
        target_modules: List of module names to adapt (if None, adapts all Linear layers)
    
    Returns:
        Dictionary mapping module names to their LoRA adapters
    """
    adapters = {}
    
    if target_modules is None:
        # Find all Linear layers
        target_modules = []
        for name, module in gemnet_model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)
    
    for module_name in target_modules:
        # Get the target module
        target_module = gemnet_model
        for part in module_name.split('.'):
            target_module = getattr(target_module, part)
        
        if isinstance(target_module, nn.Linear):
            # Create LoRA adapter
            adapter = LoRAAdapter(
                target_layer=target_module,
                rank=rank,
                alpha=alpha,
                adapter_name=f"lora_{module_name}"
            )
            adapters[module_name] = adapter
    
    return adapters


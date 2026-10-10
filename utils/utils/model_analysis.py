"""
Model analysis utilities without thop dependency
"""

import torch
import torch.nn as nn


def calculate_model_complexity_safe(model, input_size=(1, 3, 352, 352)):
    """Calculate model complexity without using thop to avoid hook conflicts"""
    model.eval()
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate GFLOPs using alternative method
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        gflops = calculate_flops_alternative(model, dummy_input)
    except Exception as e:
        print(f"Warning: Could not calculate GFLOPs: {e}")
        gflops = 0.0
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gflops': gflops,
        'params_mb': total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    }


def calculate_flops_alternative(model, dummy_input):
    """Alternative FLOPs calculation method"""
    try:
        # Simple FLOPs estimation based on parameters and input size
        total_params = sum(p.numel() for p in model.parameters())
        input_size = dummy_input.numel()
        
        # Rough estimation: each parameter corresponds to about 2-3 floating point operations
        estimated_flops = total_params * 2.5 + input_size * 10
        return estimated_flops / 1e9
    except:
        return 0.0


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_mb': total_params * 4 / (1024 * 1024)
    }


def analyze_model_layers(model):
    """Analyze model layers and their parameters"""
    layer_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params
                })
    
    return layer_info

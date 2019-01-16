import matplotlib.pyplot as plt
import torch

def decudify(obj, scalar=False):
    if issubclass(type(obj), dict):
        return {k:decudify(v, scalar=scalar) for k,v in obj.items()}
    elif issubclass(type(obj), list):
        return [decudify(i, scalar=scalar) for i in obj]
    elif issubclass(type(obj), tuple):
        return tuple([decudify(i, scalar=scalar) for i in list(obj)])
    elif torch.is_tensor(obj):
        obj_cpu = obj.cpu()
        if scalar:
            obj_cpu = obj_cpu.detach().numpy()
        del obj
        return obj_cpu
    else:
        return obj
    
    
def get_cmap(n, color_map='plasma'):
    return plt.cm.get_cmap(color_map, n)
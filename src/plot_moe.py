import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.dataset import ROLE_PROFILE_MAPPING

def load_role_profile(role_embds_path: str):
    for k in ROLE_PROFILE_MAPPING:
        role_embd = os.path.join(role_embds_path, f"{k}.pth")
        if os.path.exists(role_embd):
            ROLE_PROFILE_MAPPING[k] = torch.load(
                os.path.join(role_embds_path, f"{k}.pth"), 
                map_location="cpu"
            )
        
    keys = list(ROLE_PROFILE_MAPPING.keys())
    for k in keys:
        if not isinstance(ROLE_PROFILE_MAPPING[k], torch.Tensor):
            del ROLE_PROFILE_MAPPING[k]
    
    print(list(ROLE_PROFILE_MAPPING.keys()))

def plot_sim_matrix(sim_matrix, labels:list):
    fig, ax = plt.subplots()
    ax.imshow(sim_matrix)
    
    ax.set_xticks(range(len(labels)), labels, rotation=45, 
                  ha="right", rotation_mode = "anchor")
    ax.set_yticks(range(len(labels)), labels)
    # add text
    x, y = sim_matrix.shape
    for i in range(x):
        for j in range(y):
            ax.text(j, i, round(sim_matrix[i,j], 2), ha="center", va="center", color="w")

    fig.tight_layout()
    
    return fig, ax


def plot_embd_sim(title: str):
    # title = "role embedding sim"
    # print(ROLE_PROFILE_MAPPING)

    # print(exist_role_embds)
    role_embds_sim = torch.stack(list(ROLE_PROFILE_MAPPING.values()), dim=0).detach().numpy()
    role_embds_sim /= np.linalg.norm(role_embds_sim, axis=1, keepdims=True)
    role_embds_sim = role_embds_sim @ role_embds_sim.T

    fig, ax = plot_sim_matrix(role_embds_sim, labels = list( ROLE_PROFILE_MAPPING.keys()))
    ax.set_title(title)

    return fig, ax

def plot_expert_act(data, figsize=(10,10)):
    '''
    plot expert activation
    '''
    data_culm = data.cumsum(axis=-1)
    layers = data.shape[0]
    experts = data.shape[1]
    # labels for xaxis
    label_intvl = 5
    layer_labels = ["Layer#{}".format(i+1) for i in range(0, layers, label_intvl)]
    # colormap
    colors = plt.colormaps["tab20"](np.linspace(0.2, 0.8, data.shape[1]))
    fig, ax = plt.subplots(figsize=figsize)
    # close x axis
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    
    for i, color in enumerate(colors):
        widths = data[:, i]
        starts = data_culm[:, i] - widths
        rects = ax.barh(range(layers), widths,
                        left=starts, height=0.5, label=str(i),
                        color=color)
    ax.legend(ncols = layers, bbox_to_anchor=(0,1), loc="lower left", fontsize=8)
    ax.set_ylim(layers-0.5, -0.5)
    ax.set_yticks(list(range(0, layers, label_intvl)), layer_labels)

    return fig, ax

@torch.no_grad()
def calculate_gate_probs(gating, role_embd):
    if role_embd.dim() == 1:
        role_embd.unsqueeze_(0)
    
    logits = gating(role_embd)[1]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs


@torch.no_grad()
def plot_activation_by_role(model, role: str, module_name: str, figsize=(8,10)):
    '''
    plot the gating preference for `role` wrt each layer
    '''
    layers = model.base_model.model.model.layers
    active_adapter = model.active_adapter

    role_embd = ROLE_PROFILE_MAPPING[role]
    act = []

    for layer in layers:
        module = getattr(layer.self_attn, module_name)
        gating = module.gating[active_adapter]

        # gating now only return logits
        probs = calculate_gate_probs(gating, role_embd.to(model.device))
        act.append(probs.cpu().numpy())

    data = np.concatenate(act, axis=0)
    fig, ax=plot_expert_act(data, figsize=figsize)
    fig.suptitle("Expert act of {}/{}".format(role, module_name))
    fig.tight_layout()

    return fig, ax

@torch.no_grad()
def get_gating_for_roles(model, module_name:str):
    layers = model.base_model.model.model.layers
    n_layers = len(layers)

    gating_weights = []
    for k in ROLE_PROFILE_MAPPING:
        act = []
        role_embd = ROLE_PROFILE_MAPPING[k].to(model.device)
        for layer in layers:
            module = getattr(layer.self_attn, module_name)
            n = module.num_moe[model.active_adapter]
            r_single = module.r[model.active_adapter] // n

            # (n, 1)
            gating = module.gating[model.active_adapter]
            gating_probs = calculate_gate_probs(gating, role_embd).T

            # (d, r, n)
            loraA = module.lora_A[model.active_adapter].weight.T.reshape(-1, r_single, n)
            # (r, n, d)
            loraB = module.lora_B[model.active_adapter].weight.T.reshape(r_single, n, -1)

            # (n, d)
            # print(gating_probs.shape)
            lora_weight = (torch.einsum("drn, rnd->nd", loraA, loraB) * gating_probs).sum(0)
            lora_weight = lora_weight.cpu().numpy()
            act.append(lora_weight)

        data = np.stack(act, axis=0)
        gating_weights.append(data)
    
    layer_axis, role_axis, exp_axis = range(3)
    # print(np.stack(gating_weights, axis=role_axis).shape)

    # gating_weights = role*[layer, n_exp] => [layer, role, exp]
    gating_weights = np.split(np.stack(gating_weights, axis=role_axis), n_layers, axis=layer_axis)
    for i in range(len(gating_weights)):
        gating_weights[i] /= np.linalg.norm(gating_weights[i], axis=exp_axis, keepdims=True)
        gating_weights[i] = gating_weights[i].squeeze(layer_axis)
    
    # n_layers * [n_role, n_exp]
    return gating_weights

def plot_mean_sim(model, module_name:str):
    # layers = model.base_model.model.model.layers
    gating_weights = get_gating_for_roles(model, module_name)

    # [layer, n_role, n_role]
    # calculate similarity
    sim_matrix = np.stack([gw @ gw.T for gw in gating_weights], axis=0)
    print(sim_matrix.shape)
    fig, ax = plot_sim_matrix(sim_matrix.mean(0), list(ROLE_PROFILE_MAPPING.keys()))
    ax.set_title("{} mean similarity wrt each character".format(module_name))

    return fig, ax

def plot_tsne(gating_weight, perplexity: int = 3, **tsne_args):
    '''
    calculate gating weight with `get_gating_for_roles()`
    and pick a slice as param `gating_weight`
    '''
    tsne = TSNE(n_components=2, perplexity=perplexity, **tsne_args)
    X_tsne = tsne.fit_transform(gating_weight)
    # normalize tsne
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne-x_min)/(x_max-x_min)

    labels = list(ROLE_PROFILE_MAPPING.keys())
    
    fig, ax = plt.subplots()
    ax.scatter(X_norm[:,0], X_norm[:,1], 
               color=plt.colormaps["tab20"](np.linspace(0.2, 0.8, X_tsne.shape[0])))
    for i in range(X_norm.shape[0]):
        ax.text(X_norm[i, 0], X_norm[i,1], labels[i])

    return fig, ax
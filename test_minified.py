import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import profile
from scvi.hub import HubModel
from time import sleep
import sys

@profile
def minified_hub_model_generate():
    hubmodel = HubModel.pull_from_huggingface_hub("scvi-tools/human-lung-cell-atlas", local_dir='hlca')
    adata = hubmodel.adata
    model = hubmodel.model
    sleep(5)
    model.adata.layers['imputed'] = model.get_normalized_expression(adata, batch_size=16000)
    sleep(5) 
    
@profile
def minified_hub_model_latent():
    hubmodel = HubModel.pull_from_huggingface_hub("scvi-tools/human-lung-cell-atlas", local_dir='hlca')
    adata = hubmodel.adata
    model = hubmodel.model
    sleep(5)
    adata.obsm['scvi'] = model.get_latent_representation(batch_size=16000)
    sleep(3)
    
@profile
def usual_hub_model_generate():
    hubmodel = HubModel.pull_from_huggingface_hub("scvi-tools/human-lung-cell-atlas", local_dir='hlca')
    adata = hubmodel.large_training_adata
    scvi.model.SCANVI.prepare_query_anndata(adata, "hlca")
    model = scvi.model.SCANVI.load("hlca", adata=adata, use_gpu=True)
    sleep(5)
    model.adata.layers['imputed'] = model.get_normalized_expression(adata, batch_size=16000)
    sleep(5) 
       
@profile
def usual_hub_model_latent():
    hubmodel = HubModel.pull_from_huggingface_hub("scvi-tools/human-lung-cell-atlas", local_dir='hlca')
    adata = hubmodel.large_training_adata
    scvi.model.SCANVI.prepare_query_anndata(adata, "hlca")
    model = scvi.model.SCANVI.load("hlca", adata=adata, use_gpu=True)
    sleep(5)
    adata.obsm['scvi'] = model.get_latent_representation(batch_size=16000)
    sleep(3)

if __name__ == '__main__':
    globals()[sys.argv[1]]()
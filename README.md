# Scvi-hub: an actionable repository for model-driven single cell analysis

This repository contains notebooks and scripts to reproduce analyses benchmarking the use of scvi-hub repository models for single-cell genomic analysis.
(see [manuscript](https://www.biorxiv.org/content/10.1101/2024.03.01.582887v1)).

We provide HuggingFace scvi-hub models at [HuggingFace](https://huggingface.co/scvi-tools) and recommend reviewing the [scvi-tools](https://scvi-tools.org/) documentation to get started. Especially [CELLxGENE hub](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/hub/cellxgene_census_model.html)
highlights how to use the CELLxGENE census reference model.

## Repository structure
We provide here all notebooks to reproduce the results. Data and reference models are pulled from public repositories on 03/10/24. The census soma object
that was used in figure 4 and 5 can be shared upon request. However, similar results can be produced using latter

- analysis notebooks and scripts:
  - `suppl_fig1_minified_benchmark.ipynb` - notebook to recreate benchmarking memory usage (not included in the manuscript)
  - `suppl_fig1_profile_memory.ipynb` - notebook to recreate benchmarking in Supplementary Figure 1 of the manuscript.
  - `fig2_hlca_only_reference.ipynb` - notebook to recreate figure 2 of the manuscript in which we focus on the HLCA and validate the use of a minified model. In addition, code to reproduce Supplementary Figure 2 is contained.
  - `suppl_fig3_4_heart_cell_atlas_criticism.ipynb` - Validation of criticism metrics to detect overfit and poorly trained models. Detect not appropriate reference model for query data (lung model for modelling epithelial cells in other organs). Supplementary Figure 3 and 4.
  - `fig3_hlca_emphysema.ipynb` - notebook to recreate figure 3A-F of the manuscript in which we focus on an emphysema dataset using the HLCA as the reference model. We discover functional states of fibroblasts associated with lung emphysema. In addition, code to reproduce Supplementary Figure 5 and 6 is contained.
  - `fig3_hlca_infusion.ipynb` - notebook to recreate figure 3G-J of the manuscript in which we focus on injecting labels from a cross-tissue immune cell atlas to the HLCA and perform differential expression analysis of CD8 TRM in COVID. In addition, code to reproduce Supplementary Figure 10 and 11 is contained.
  - `suppl_fig9_deconvolution_prostate.ipynb` - notebook to perform deconvolution analysis of prostate tissue (Supplementary Figure 9).
  - `fig4_5_census_cart_query.ipynb` - notebook to fetch and train census model and recreate figure 4 and 5. Map CAR-T cells to reference model and transfer labels to these cells. Perform criticism metrics on this query dataset and subclustering analysis. Transfer labels from cross-organ immune cell atlas to CELLxGENE census cells and perform fine analysis of all TREGS including subclustering and DE analysis. In addition, code to reproduce Supplementary Figure 12, 14, 15 is contained.
  - `suppl_fig13_census_cart_query_panimmune.ipynb` - Analysis to perform query mapping of CAR-T cells to cross-organ immune cell reference atlas. Supplementary Figure 13.
  - `suppl_fig16_totalvi_criticism.ipynb` - Analysis of COVID CITE-seq data and criticism metric. Supplementary Figure 13.

## Citation

> Scvi-hub: an actionable repository for model-driven single cell analysis. Can Ergen, Valeh Valiollah Pour Amiri, Martin Kim, Aaron Streets, Adam Gayoso, Nir Yosef. bioRxiv 2024.03.01.582887; doi: https://doi.org/10.1101/2024.03.01.582887.

## Contact

For any questions, please post an [issue](https://github.com/YosefLab/scvi-hub-reproducibility/issues) or reach out on [discourse](https://discourse.scverse.org).
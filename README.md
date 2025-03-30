# GenCast - Utilizing Sparse Transformers within Machine Learning Weather Prediction (MLWP)
Price, I., Sanchez-Gonzalez, A., Alet, F. et al. Probabilistic weather forecasting with machine learning. Nature 637, 84–90 (2025). https://doi.org/10.1038/s41586-024-08252-9

## Overview

## Question:
Why would a sparse transformer be preferred to a normal transformer?

- Answer: Compute and also physics - Performing attention for every node on all others is a huge load on the model. Additionally, we expect the weather in Nashville to be primarily affected by conditions in our local area e.g. Tennessee, rather than the conditions in Australia for example.

## Architecture Overview




## Critical Analysis

## Impacts
- This work as well as GraphCast have introduced MLWP as a capable tool for decisionmakers globally. While NWP is still greatly needed, MLWP models such as these can provide rapid cross validation and data to stakeholders around extreme weather events.
- One author on both (Ferran Alet) has spoken about how this work has alerted European weather prediction agencies to the power of MLWP and are now investing in GPU infrastructure for this purpose

## Resources
- [Ferran Alet speaking on: Graph Neural Networks for Skillful Weather Forecasting](https://youtu.be/ez1pIFcU52s?si=37FJSf73FI5CInzn)
- GenCast arXiv citation:
  - Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., & Willson, M. (2023), “GenCast: Diffusion-based ensemble forecasting for medium-range weather”, Nature,  [https://doi.org/10.48550/arXiv.2312.15796](https://doi.org/10.1038/s41586-024-08252-9), arXiv:2312.15796.
- GraphCast Paper:
  - Lam, R., Sanchez-Gonzalez, A., Willson, M., Wirnsberger, P., Fortunato, M., Alet, F., Ravuri, S., Ewalds, T., Eaton-Rosen, Z., Hu, W., Merose, A., Hoyer, S., Holland, G., Vinyals, O., Stott, J., Pritzel, A., Mohamed, S., & Battaglia, P. (2023), “Learning skillful medium-range global weather forecasting”, Science, Volume 382, Issue 6677, pp. 1416-1421 (2023)., 382, 1416-1421 (6 pp),  [https://doi.org/10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336), arXiv:2212.12794.
- Understanding Diffusion Models + tuning them:
  - Karras, T., Aittala, M., Aila, T., & Laine, S. (2022), “Elucidating the Design Space of Diffusion-Based Generative Models”, eprint arXiv:2206.00364, arXiv:2206.00364 ( pp),  [https://doi.org/10.48550/arXiv.2206.00364](https://doi.org/10.48550/arXiv.2206.00364), arXiv:2206.00364.
- [Github Repo for GenCast and GraphCast](https://github.com/google-deepmind/graphcast)
- [Google DeepMind Blog Post about GenCast](https://deepmind.google/discover/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/)
- [ECMWF Reanalysis v5 {ERA5} - the dataset GenCast was trained on](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)






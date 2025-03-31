# GenCast - Utilizing Sparse Transformers within Machine Learning Weather Prediction (MLWP)
Price, I., Sanchez-Gonzalez, A., Alet, F. et al. Probabilistic weather forecasting with machine learning. Nature 637, 84–90 (2025). https://doi.org/10.1038/s41586-024-08252-9

## Overview
- Individuals, businesses, and governments around the world use weather predictions to make operational decisions, affecting billions of lives. These weather predictions are produced via **Numerical Weather Prediction (NWP)** wherein supercomputers programmed with the laws of physics and atmospheric chemistry take in observations of current atmospheric conditions and forward model their time evolution over the next 10-15 days.
- Since weather is inherently chaotic, it is impossible to predict the exact weather state of the future. NWP accounts for this by introducing perturbations of initial conditions and sub-resolution phenomena or combining multiple models **to generate an ensemble of weather forecasts, encompassing the probability distribution of future weather**.
  - This distribution becomes narrower for shorter prediction time periods (we are more certain about the weather 1 day from now versus 10 days from now).
  - NWP accuracy scales well with compute but has no ability to learn from past weather.
- **Machine Learning Weather Prediction (MLWP)** has recently emerged using a variety of techniques (Convolutional Neural Networks, Graph Neural Networks, Fourier Neural Operators, and Transformers -- see citations for works). These approaches have not been probalistic in nature, however, usually forecasting the _mean_ of forecast trajectories. **Without the probabilistic nature of the forecast, these methods are less useful for stakeholders.** 
  - Additionally, they tend to "blur" predictions with long lead times (10 days out) as this minimizes the error but it's predictions are not physically real versions of weather.

- GenCast solves these problems by 

## Architecture Overview


Sparse Transformer
- easier to compute (less connections for each node)
- inherently account for geography of earth



![image](./images/graphcastmesh.png) 

![image](./images/sparsetrans.png) 


## Question:
Why would a sparse transformer be preferred to a normal transformer?

- Answer: Compute and also physics - Performing attention for every node on all others is a huge load on the model. Additionally, we expect the weather in Nashville to be primarily affected by conditions in our local area e.g. Tennessee, rather than the conditions in Australia for example.

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
- Other MLWP
  - Transformers: 
    - Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2022), “Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast”, eprint arXiv:2211.02556, arXiv:2211.02556 ( pp),  [https://doi.org/10.48550/arXiv.2211.02556](https://doi.org/10.48550/arXiv.2211.02556), arXiv:2211.02556.
    - Nguyen, T., Brandstetter, J., Kapoor, A., Gupta, J. K., & Grover, A. (2023), “ClimaX: A Foundation Model for Weather and Climate”, AGU Fall Meeting 2023, held in San Francisco, CA, 11-15 December 2023, Session: Global Environmental Change / Deep Learning in Climate, Weather, and Earth Sciences I Oral, id. GC21A-01., 2023, GC21A-01 ( pp),  [https://doi.org/10.48550/arXiv.2301.10343](https://doi.org/10.48550/arXiv.2301.10343), arXiv:2301.10343.
    - Chen, K., Han, T., Gong, J., Bai, L., Ling, F., Luo, J.-J., Chen, X., Ma, L., Zhang, T., Su, R., Ci, Y., Li, B., Yang, X., & Ouyang, W. (2023), “FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead”, eprint arXiv:2304.02948, arXiv:2304.02948 ( pp),  [https://doi.org/10.48550/arXiv.2304.02948](https://doi.org/10.48550/arXiv.2304.02948), arXiv:2304.02948.
  - Fourier Neural Operators
    - Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A., Mardani, M., Kurth, T., Hall, D., Li, Z., Azizzadenesheli, K., Hassanzadeh, P., Kashinath, K., & Anandkumar, A. (2022), “FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators”, eprint arXiv:2202.11214, arXiv:2202.11214 ( pp),  [https://doi.org/10.48550/arXiv.2202.11214](https://doi.org/10.48550/arXiv.2202.11214), arXiv:2202.11214.
  - Graph Neural Networks
    - Keisler, R. (2022), “Forecasting Global Weather with Graph Neural Networks”, eprint arXiv:2202.07575, arXiv:2202.07575 ( pp),  [https://doi.org/10.48550/arXiv.2202.07575](https://doi.org/10.48550/arXiv.2202.07575), arXiv:2202.07575.
  - Convolutional Neural Networks
    - Weyn, J. A., Durran, D. R., &
Caruana, R. (2019). Can machines
learn to predict weather? Using deep
learning to predict gridded 500-hPa
geopotential height from historical
weather data. Journal of Advances
in Modeling Earth Systems, 11,
2680–2693. https://doi.org/10.1029/
2019MS001705
    - Weyn, J. A., Durran, D. R., Caruana, R., & Cresswell-Clay, N. (2021), “Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models”, Journal of Advances in Modeling Earth Systems, Volume 13, Issue 7, article id. e2021MS002502, 13, e2021MS002502 ( pp),  [https://doi.org/10.1029/2021MS002502](https://doi.org/10.1029/2021MS002502), arXiv:2102.05107.
    - Rasp, S., & Thuerey, N. (2021), “Data Driven Medium Range Weather Prediction With a Resnet Pretrained on Climate Simulations: A New Model for WeatherBench”, Journal of Advances in Modeling Earth Systems, Volume 13, Issue 2, article id. e2020MS002405, 13, e2020MS002405 ( pp),  [https://doi.org/10.1029/2020MS002405](https://doi.org/10.1029/2020MS002405), arXiv:2008.08626.















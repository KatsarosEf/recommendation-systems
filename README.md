# collaborative-filtering-recommendation-systems
This repo contains the code and report for the "Advances in Data Mining" course, at Leiden University for the academic year 2018-2019. The scripts implement different models on collaborative filtering. The dataset used is the [1M version of movielens](
https://grouplens.org/datasets/movielens/?fbclid=IwAR05s5xjdWm9vEAPinrCuWwI5gDlR-JPqtpPYsREKoFmmoYXCBQ8LwGcl-k). The rating database contains 6k users and 4k movies. More information and elaborative experiments can be found on the first section of the attached report.


## Baseline models
Predict the rating using a) the global mean b) items mean c) users mean d) weight average of item and user mean values. We observe their performance based on the RMSE and MAE metrics.

## Matrix factorization (Stochastic Gradient Descent)
We implement Matrix factorization, a top-performing model on the Netflix prize competition. We resort to Stochastic Gradient Descent for optimization. Moreover, we implement and discuss further experiments on the hyper-parameters.

## Matrix factorization (Alternating Least Squares)
The MF model is now optimized with the Alternating Least Squates. Likewise, it converges more than ten times faster with slightly performance

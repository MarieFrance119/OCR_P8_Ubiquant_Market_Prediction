# OCR_P8_Ubiquant_Market_Prediction
[Ubiquant Market Prediction](https://www.kaggle.com/competitions/ubiquant-market-prediction/overview) is a Kaggle competition to which I am participating. It began on January, 18, 2022 and will end on April 18, 2022.

This is prediction competition through regression.

2 notebooks have been written, one for analysis and one for modelisation.

I tested the influence of preprocessing on modelisation, considering scaling and dimension reduction. I focused only on the 300 features.

Models used in the notebook of modelisation are :
- LinearRegression
- ElasticNet
- XGBRegressor
- unidimensionnal convolutionnal neuronal networks (Conv1D).

Several architectures for Conv1D have been tested with different activation functions.

In this study, best model is Conv1D. Nevertheless, Pearson correlation coefficient (between actual and predicted) are low (around 0.1). LinearRegression, even it is a simple model, does not so bad comparing others. 

Two kernels have been published on Kaggle :
- [one](https://www.kaggle.com/code/larochemf/ubiquant-low-memory-use-be-careful) about problems caused when changing the data type to reduce the size (in bytes) of the dataset 
- [one](https://www.kaggle.com/code/larochemf/eda-pca-linearregression/notebook) about exploratory analysis, PCA and LinearRegression

# Time Series Forecasting with Moirai Moirai is a time series language model from Salesforce. Named for the
three fates who control destiny, it is easy-to-use and fast (wall...

### Time Series Forecasting with Moirai (an LLM transformer)

Moirai is a time series language model from Salesforce. Named for the three fates who control destiny, it is easy-to-use and fast (wall time was 1.7 seconds for this to make the predictions below). Moirai is a Masked Encoder-based Universal Time Series Forecasting Transformer trained on the Large-scale Open Time Series Archive (LOTSA) \[27 billion observations across nine domains\].

The goal of Moirai is to make a generalizable model. It does this by focusing on cross-frequency learning and using varying distributional properties in large datasets. As a result, it can handle an arbitrary numbers of variates in multivariate time series with a zero-shot forecaster.

The dataset contains hourly load values from ERCOT, the grid balancing authority in Texas. Moirai expects the data to be in GluonTS format, so we convert from dataframe to that.


### Train-Test Split
We split the dataset into training and testing portions. The test set contains the last 64 hours.


### Model Setup
Moirai is a large-scale time series transformer. We load a pre-trained version and configure it for ERCOT data.


We create a predictor and generate forecasts.


### Visualization
We extract one example and plot the results.



Moirai forecasts ERCOT load with minimal setup. It has wide bands of uncertainty but overall, it is a nice framework. I revisited Moirai to see how it did with other generate data. Below is the new code using [EIA generation data.](https://www.eia.gov/electricity/data/browser/)

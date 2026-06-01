# Time Series Forecasting with Moirai

This project demonstrates time series forecasting using the Moirai model from Salesforce.

## Business context

Moirai is a time series language model from Salesforce. Named for the three fates who control destiny, it is easy-to-use and fast (wall time was 1.7 seconds for this to make the predictions below). Moirai is a Masked Encoder-based Universal Time Series Forecasting Transformer trained on the Large-scale Open Time Series Archive (LOTSA) [27 billion observations across nine domains].

The goal of Moirai is to make a generalizable model. It does this by focusing on cross-frequency learning and using varying distributional properties in large datasets. As a result, it can handle an arbitrary numbers of variates in multivariate time series with a zero-shot forecaster.

The dataset contains hourly load values from ERCOT, the grid balancing authority in Texas. Moirai expects the data to be in GluonTS format, so we convert from dataframe to that.

## Article

Medium article: [Time Series Forecasting with Moirai](https://medium.com/@kylejones_47003/time-series-forecasting-with-moirai-5d063a40e9ed)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Moirai forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and column names
- Model parameters (size, prediction_length, context_length)
- Test set size
- Output settings

## Moirai Model

Moirai is a foundation model for time series forecasting:
- Pre-trained on diverse time series
- Supports multiple forecast horizons
- Handles missing values and irregular frequencies
- Uses transformer architecture

## Caveats

- Requires data file with date and value columns.
- Model downloads pre-trained weights on first run (requires internet).
- GPU recommended for faster inference but not required.
- Prediction length should match test_size for proper evaluation.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
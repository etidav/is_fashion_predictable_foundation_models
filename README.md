# is fashion predictable? part 2 
Code repository to reproduce results of the Medium article ["Is Fashion Predictable? The rise of foundation models"]()

## Code Organisation

The repository is organized as follows:

 - [statsforecast_model.py](src/is_fashion_predictable_foundation_models/model/foundation_model.py): File with all the foundation models tested in the artical.
 - [metrics.py](src/is_fashion_predictable_foundation_models/metrics.py): File with the three error metrics: MASE, SMAPE and MAPE.
 - [main.py](src/is_fashion_predictable_foundation_models/main.py): File reproducing the main experience of the media article.
# Metric learning model with Triplet Loss and Hard Negative Mining

Based on [Olivier Moindrot blog post](https://omoindrot.github.io/triplet-loss)

Dataset: Stanford Online Products

How to use:

For more convenient experimenting use Docker and docker-compose to start scripts.

* Tweak config depending on the task to be solved
* Train embedding model using `train_model.py`
* Evaluate model using appropriate _k_ in `evaluate.py`
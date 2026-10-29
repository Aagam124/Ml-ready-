"""The MLReady analysis engine: plugin checks -> findings -> one explainable score.

Pure pandas / NumPy / SciPy / scikit-learn. No web framework and no database imports,
so it can be unit-tested and reused by the worker, the API and the exported scripts.
"""

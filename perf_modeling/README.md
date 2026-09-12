# Performance modeling

Once `serving/results/vllm_bench.csv` and `sglang_bench.csv` have real rows (see [`../serving/`](../serving/)), train two small models that predict throughput and p99 latency from `(engine, batch_size, input_len, output_len)`:

```bash
pip install -r requirements.txt

python train_sklearn_regressor.py   # GradientBoostingRegressor baseline
python train_tf_model.py            # small Keras MLP
python compare_models.py            # side-by-side MAE on a held-out split

# or predict a specific, possibly-unseen configuration:
python compare_models.py --engine vllm --batch-size 16 --input-len 512 --output-len 128
```

`data.py` is shared by both training scripts so they train on the exact same feature encoding - otherwise the comparison in `compare_models.py` wouldn't be apples-to-apples.

Why both a scikit-learn *and* a TensorFlow model on the same small tabular dataset: it's an honest test of whether the deep-learning model actually earns its complexity here, rather than assuming it does. `compare_models.py`'s MAE table is the answer, not an assumption.

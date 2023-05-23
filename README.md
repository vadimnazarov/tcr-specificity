# T-cell receptor specificity prediction via Deep Learning

Use more complex approaches (metric learning, triplet learning, cosine distance) to the prediction of T-cell specificity.

Working code:
```
python3 run_model.py -p -e 300 --mode triplet -f 64 -x 1000 --lrate 0.001 --batch 16
```

Packages:

```
torch
pandas
numpy
matplotlib
networkx
```
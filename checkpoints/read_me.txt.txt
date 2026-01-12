From epoch 1 - epoch 5 I was calculating metrics (precision, dice) on a batch level rather
than per tile. This can heavily skew metrics towards 0 on a strongly imbalanced dataset.
From epoch 6 onwards I've calculated metrics per tile.

From epoch 11 - 15, L1 loss for distance maps weight was increased to 0.4 (from 0.3).

Need to fix checkpoint epoch numbers in trainingData dict.
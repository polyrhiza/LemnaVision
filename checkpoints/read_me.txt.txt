From epoch 1 - epoch 5 I was calculating metrics (precision, dice) on a batch level rather
than per tile. This can heavily skew metrics towards 0 on a strongly imbalanced dataset.
From epoch 6 onwards I've calculated metrics per tile.
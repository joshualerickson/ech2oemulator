#!/usr/bin/env python3
"""Train a stateful Oct--Sep ConvGRU or ConvLSTM water-year experiment.

This stable generic entry point delegates to the original implementation kept
for backward compatibility.  Select the cell with ``--recurrent-cell``.
"""
from train_water_year_lstm import main


if __name__ == "__main__":
    main()

# TODO

## Important: re-code the training process because all the models to learn global context have low results on the test set
## Find out why CNN is less affected by the current training process.

1. Dataset split

- Divided by partition (0, 1, 2)
- 0 and 1 are used for training and validation
    + 90% is used for training
    + 10% is used for validation
- 2 is used for testing
- 25% of dataset is used for evaluation (**need to research that how they defined this D<sub>B</sub> set**)
    - use some of the data for benchmarking or use `benchmark_set_sp5.fasta`

2. Tokenizer

- BPE Tokenizer is used (**How to build my own tokenizer and not depend on `transformers` from `hugging face`**)
- Dataset used for training tokenizer is UniProt100 with Amino Acid Smiles (**Is there any knowledge showing the best
  SMILES format, or do I need to adjust on my own?**)

3. Model

- CNN
- BiLSTM
- Transformers

4. Training
5. Evaluation

- Evaluation on different types of life groups or entire dataset?
- In the origin paper TSignal, they evaluate each life group, with F1 score and MCC

6. Visualization


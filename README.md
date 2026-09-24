# PurGE

PurGE is a hyperparameter optimization framework based on Grammatical Evolution (GE) and automated search-space pruning.

PurGE uses a two-stage optimization process. Stage 1 explores the hyperparameter search space using GE and collects the performance of the evaluated configurations. The results are analysed using statistical methods to identify relevant hyperparameters, interactions between hyperparameters, and high-performing regions of the search space. These results are used to automatically generate a pruned BNF grammar. Stage 2 applies GE to the pruned search space to obtain the final hyperparameter configuration.

Experiments with machine learning and deep learning models showed that PurGE achieved competitive or improved performance compared with Random Search, Grid Search, and Bayesian optimization. The original study reported an average computational speed-up of **47×** and a reduction of **28–35% in the number of trials**. For the complete methodology and experimental evaluation, please refer to the [publications](#publications).

---

## 1. Grammatical Evolution

Grammatical Evolution (GE) is an evolutionary computation technique for generating solutions using a grammar [(O’Neill and Ryan)](https://link.springer.com/book/10.1007/978-1-4615-0447-4). Candidate solutions are represented by a genome consisting of codons and mapped to a phenotype using a Backus–Naur Form (BNF) grammar.

The grammar defines the possible solutions that can be generated during evolution. A fitness function evaluates each generated solution, and the evolutionary process searches for solutions that maximise or minimise the defined objective.

In PurGE, GE is used to generate and evaluate hyperparameter configurations. The BNF grammar defines the hyperparameters and their possible values.

---

## 2. Hyperparameter Optimization

Hyperparameter optimization (HPO) identifies a set of hyperparameters that optimises the performance of a machine learning or deep learning model for a given dataset.

The computational cost of HPO increases with the number of hyperparameters, the number of possible values, and the cost of training and evaluating the model. PurGE addresses this computational requirement by identifying high-performing regions of the hyperparameter search space and using them to construct a reduced search space for further optimization.

---

## 3. PonyGE2

PurGE is implemented using PonyGE2, a Python implementation of Grammatical Evolution.

PonyGE2 provides the underlying GE functionality, including genotype-to-phenotype mapping, population initialization, selection, crossover, mutation, and fitness evaluation.

PurGE extends this implementation with the two-stage HPO and search-space pruning procedure.

PonyGE2 reference:
https://arxiv.org/abs/1703.08535

---

## 4. PurGE

The PurGE framework consists of **Stage 1, search-space pruning, and Stage 2**.

The complete architecture of PurGE is presented in **Figure 2 of the original PurGE publication**.

<img width="357" height="163" alt="download" src="https://github.com/user-attachments/assets/0b2cb9fa-c163-433c-b8b9-7e1cfeabff39" />
> **Figure 2:** Architecture of PurGE, a two-stage Grammatical Evolution-driven approach for automatically evolving hyperparameters with search-space pruning.
> See Vaidya, Kshirsagar, and Ryan (2025).

### Stage 1: Search-Space Exploration

Stage 1 applies GE to the complete hyperparameter search space represented by the initial BNF grammar.

Each trial evaluates one hyperparameter configuration for a given model and dataset. The resulting configurations and their validation performance are recorded for the pruning analysis.

In the original study, Stage 1 used **60% of the total trial budget**.

### Search-Space Pruning

The results generated during Stage 1 are analysed to identify high-performing regions of the hyperparameter search space.

The pruning procedure uses:

* **Pearson correlation analysis** to measure relationships between individual hyperparameters and validation accuracy;
* **inter-hyperparameter correlation analysis** to identify dependencies between pairs of hyperparameters; and
* **Individual Conditional Expectation (ICE)** analysis to identify high-performing values or ranges.

The analysis produces two sets:

* **H1:** hyperparameters associated with validation performance and their promising ranges;
* **H2:** relationships between pairs of hyperparameters used to refine the ranges identified in H1.

The resulting ranges define the reduced search space. PurGE automatically generates a new BNF grammar containing the retained hyperparameter values.

### Stage 2: Optimization Within the Pruned Search Space

Stage 2 applies GE to the pruned BNF grammar.

The configurations generated from the reduced search space are evaluated using the same objective function. The optimization continues within this search space until the allocated trial budget is reached.

The final output is the hyperparameter configuration with the best validation performance.

For the complete algorithm, equations, experimental design, and evaluation, refer to the [publications](#publications).

---

## 5. Requirements

PurGE requires Python 3.5 or higher.

The main dependencies are:

* NumPy
* SciPy
* pandas
* scikit-learn

All required packages are listed in `requirements.txt`.

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

---

## 6. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── datasets/
│   └── Banknote/
├── grammars/
│   └── purge/
│       ├── rf_banknote.bnf
│       └── generated/
├── parameters/
│   └── purge_base.txt
├── src/
│   ├── ponyge.py
│   └── fitness/
│       └── purge_hpo.py
└── experiments/
    └── purge/
        ├── purge_pipeline.py
        ├── purge_prune.py
        ├── README.md
        └── results/
            ├── runs_index.csv
            └── <dataset>/<run_id>/
                ├── stage1_trials.csv
                ├── stage2_trials.csv
                ├── pruned_grammar.bnf
                └── manifest.json
```

The main components are:

* `grammars/purge/` — BNF grammars defining the hyperparameter search spaces.
* `grammars/purge/generated/` — automatically generated pruned grammars.
* `parameters/purge_base.txt` — GE configuration parameters.
* `src/fitness/purge_hpo.py` — model evaluation and trial logging.
* `experiments/purge/purge_pipeline.py` — execution of Stage 1, pruning, and Stage 2.
* `experiments/purge/purge_prune.py` — statistical analysis and grammar pruning.
* `experiments/purge/results/` — outputs generated by PurGE experiments.

The underlying GE implementation also generates a `results/` directory at the repository root containing GE run information and generation-level statistics.

---

## 7. Running PurGE

A working example is provided using a `RandomForestClassifier` and the Banknote dataset.

Run the complete pipeline from the repository root:

```bash
python experiments/purge/purge_pipeline.py
```

The example performs:

```text
Stage 1
   ↓
Collect trial results
   ↓
Analyse hyperparameters
   ↓
Prune search space
   ↓
Generate pruned BNF grammar
   ↓
Stage 2
   ↓
Final hyperparameter configuration
```

The current example uses **80 trials**:

```text
Stage 1: 50 trials
Stage 2: 30 trials
```

The experimental settings used in the published studies are described in the corresponding publications.

---

## 8. Output

The PurGE pipeline stores results in:

```text
experiments/purge/results/
```

Each run produces:

```text
<dataset>/<run_id>/
├── stage1_trials.csv
├── stage2_trials.csv
├── pruned_grammar.bnf
└── manifest.json
```

`stage1_trials.csv` contains the hyperparameter configurations and performance values generated during Stage 1.

`stage2_trials.csv` contains the configurations evaluated using the pruned search space.

`pruned_grammar.bnf` contains the BNF grammar generated by the pruning procedure.

`manifest.json` contains metadata associated with the run.

`runs_index.csv` provides an index of the completed runs.

---

## 9. Using PurGE With a Different Model or Dataset

PurGE can be configured for another model or dataset by defining:

1. a BNF grammar containing the hyperparameter search space;
2. a fitness function for training and evaluating the model;
3. the dataset; and
4. the corresponding GE parameter configuration.

### BNF Grammar

The grammar defines the hyperparameters and values available to GE.

For example:

```bnf
<hyperparams> ::= {"'C'": <C>, "'penalty'": <penalty>}
<C> ::= 0.01 | 0.1 | 1.0 | 10.0 | 100.0
<penalty> ::= "'l1'" | "'l2'"
```

Grammars used by PurGE are stored under:

```text
grammars/purge/
```

### Fitness Function

The fitness function defines how each generated hyperparameter configuration is evaluated.

The implementation in:

```text
src/fitness/purge_hpo.py
```

can be used as the reference implementation when adding a new model.

The fitness function must also record the evaluated configurations and their performance so that the Stage 1 results can be used by the pruning procedure.

### Parameter Configuration

The base GE configuration is provided in:

```text
parameters/purge_base.txt
```

Create or modify a parameter file with the required fitness function, dataset paths, grammar, and GE settings.

### Pipeline Configuration

The model, dataset, grammar, and parameter configuration used by the pipeline are specified in:

```text
experiments/purge/purge_pipeline.py
```

After configuring these components, run:

```bash
python experiments/purge/purge_pipeline.py
```

Detailed implementation notes are available in:

```text
experiments/purge/README.md
```

---

## 10. Publications

The complete PurGE methodology and experimental evaluation are presented in the following publications.

### PurGE

Gauri Vaidya, Meghana Kshirsagar, and Conor Ryan.
**PurGE: Towards Responsible Artificial Intelligence Through Sustainable Hyperparameter Optimization.**
*Proceedings of the 17th International Conference on Agents and Artificial Intelligence (ICAART 2025), Volume 2*, pp. 622–633, 2025.

DOI: `10.5220/0013262100003890`

```bibtex
@inproceedings{vaidya2025purge,
  author    = {Vaidya, Gauri and Kshirsagar, Meghana and Ryan, Conor},
  title     = {{PurGE}: Towards Responsible Artificial Intelligence Through
               Sustainable Hyperparameter Optimization},
  booktitle = {Proceedings of the 17th International Conference on Agents
               and Artificial Intelligence (ICAART 2025), Volume 2},
  pages     = {622--633},
  year      = {2025},
  publisher = {SciTePress},
  doi       = {10.5220/0013262100003890},
  isbn      = {978-989-758-737-5},
  issn      = {2184-433X}
}
```

### Extended Work

Gauri Vaidya, Meghana Kshirsagar, and Conor Ryan.
**Resource-Efficient Techniques for Hyperparameter Optimization in Machine Learning.**
*Agents and Artificial Intelligence — ICAART 2025, Revised Selected Papers, Part III*, Lecture Notes in Artificial Intelligence, Vol. 16518, Springer, 2027.

```bibtex
@incollection{vaidya2027resource,
  author    = {Vaidya, Gauri and Kshirsagar, Meghana and Ryan, Conor},
  title     = {Resource-Efficient Techniques for Hyperparameter Optimization
               in Machine Learning},
  booktitle = {Agents and Artificial Intelligence. ICAART 2025.
               Revised Selected Papers, Part III},
  editor    = {van den Herik, H. Jaap and Rocha, Ana Paula and Steels, Luc},
  series    = {Lecture Notes in Artificial Intelligence},
  volume    = {16518},
  pages     = {77--87},
  year      = {2027},
  publisher = {Springer},
  address   = {Cham}
}
```

---

## 11. Funding

This work was conducted with the financial support of **Taighde Éireann — Research Ireland** under Grant No. **18/CRT/6223**.

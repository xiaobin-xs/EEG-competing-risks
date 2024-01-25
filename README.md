# Neurological Prognostication of Post-Cardiac-Arrest Coma Patients Using EEG Data: A Dynamic Survival Analysis Framework with Competing Risks

This repository contains the demonstration code for the manuscript:

Shen, Xiaobin, Jonathan Elmer, and George H. Chen. "Neurological Prognostication of Post-Cardiac-Arrest Coma Patients Using EEG Data: A Dynamic Survival Analysis Framework with Competing Risks." *Machine Learning for Healthcare Conference*. PMLR, 2023.

Note that the dynamic survival analysis framework we proposed is compatible with any dynamic competing risks (DCR) models, and we include the code to train and evaluate three DCR models discussed in the paper.

### DCR Models

- Fine and Gray competing risks model, 1999.
  - R code for running the model: [`cif-fine-and-gray.Rmd`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/cif-fine-and-gray.Rmd)
- Dynamic-DeepHit, 2019
  - Original [implementation](https://github.com/chl8856/Dynamic-DeepHit) by the authors in Tensorflow; PyTorch implementation available [here](https://github.com/Jeanselme/DynamicDeepHit)
  - Our implementation combines the two versions, see files under the directory [`scr/`](https://github.com/xiaobin-xs/EEG-competing-risks/tree/master/scr)
  - To see how to train and evaluate the model with single event, two competing risks, and three competing risks, refer to [`DDH-Torch-cr1.ipynb`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/DDH-Torch-cr1.ipynb), [`DDH-Torch-cr2.ipynb`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/DDH-Torch-cr2.ipynb), [`DDH-Torch-cr3.ipynb`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/DDH-Torch-cr3.ipynb), respectively
  - To get the visualization in the manuscript, see [`eval-ddh.ipynb`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/eval-ddh.ipynb)
- dynamic deep recurrent survival analysis (DDRSA), 2022
  - No implementation by the original author; see [`ddrsa.py`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/scr/ddrsa.py) for our implementation of the model architecture
  - To  see how to train and evaluate the model with three competing risks, refer to [`ddrsa-cr3.ipynb`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/ddrsa-cr3.ipynb) and [`eval-ddrsa.ipynb`](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/eval-ddrsa.ipynb)


To-Do:

- add sample data


# Neurological Prognostication of Post-Cardiac-Arrest Coma Patients Using EEG Data: A Dynamic Survival Analysis Framework with Competing Risks
This repository contains the code for the manuscript *Neurological Prognostication of Post-Cardiac-Arrest Coma Patients Using EEG Data: A Dynamic Survival Analysis Framework with Competing Risks*.

The dir structure is largely adpated from the repo for the [PyTorch implementation](https://github.com/Jeanselme/DynamicDeepHit) of the [Dynamic-DeepHit](https://ieeexplore.ieee.org/document/8681104) model.

### Baselines
- Fine and Gray competing risks model, 1999.
    - R code for running the model: [cif-fine-and-gray.Rmd](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/cif-fine-and-gray.Rmd)
- Dynamic-FeepHit, 2019
    - Original implementation in Tensorflow; PyTorch implementation available [here](https://github.com/Jeanselme/DynamicDeepHit)
    - Our implementation combines the two versions, see files under the directory [ddh/](https://github.com/xiaobin-xs/EEG-competing-risks/tree/master/ddh)
    - To train and evaluate the model with single event, two competing risks, and three competing risks, see [DDH-Torch-cr1.ipynb](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/DDH-Torch-cr1.ipynb), [DDH-Torch-cr2.ipynb](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/DDH-Torch-cr2.ipynb), [DDH-Torch-cr3.ipynb](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/DDH-Torch-cr3.ipynb)
    - To get the visualization in the manuscript, see [examples/eval-ddh.ipynb](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/eval-ddh.ipynb)
- dynamic deep recurrent survival analysis (DDRSA), 2022
    - No implementation by the original author; see [ddh/ddrsa.py](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/ddh/ddrsa.py) for our implementation of the model architecture
    - To train and evaluate the model with three competing risks, see [ddrsa-cr3.ipynb](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/ddrsa-cr3.ipynb) and [eval-ddrsa.ipynb](https://github.com/xiaobin-xs/EEG-competing-risks/blob/master/examples/eval-ddrsa.ipynb)

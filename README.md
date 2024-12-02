# It is not a piece of cake for GPT: Explaining Textual Entailment Recognition in the presence of Figurative Language

This repository contains the code for the paper [It is not a piece of cake for GPT: Explaining Textual Entailment Recognition in the presence of Figurative Language](#), to appear at COLING 2025.

It includes the Python package to train and test the LLMs we fine-tuned, as well as the code to reproduce the approaches and experiments discussed in the paper.


## Installation
The following command will clone the project:
```
git clone https://github.com/gallipoligiuseppe/FigurativeTER-E_LLM.git
```

To install the required libraries and dependencies, you can refer to the `env.yml` file.

Before experimenting, you can create a virtual environment for the project using Conda.
```
conda create -n figurative_llm -f env.yml
conda activate figurative_llm
```

The installation should also cover all the dependencies. If you find any missing dependency, please let us know by opening an issue.

## Usage

The package provides the scripts to implement the different approaches discussed in the paper.

### Data

The [FLUTE](https://aclanthology.org/2022.emnlp-main.481/) benchmark dataset can be downloaded from the official repository [page](https://github.com/tuhinjubcse/model-in-the-loop-fig-lang). Please put it into the `data` directory.

[DREAM](https://aclanthology.org/2022.naacl-main.82/) scene elaborations of FLUTE data can be extracted using the `DreamFlute.ipynb` notebook.
Pre-computed outputs will be made available upon verifying the dataset license.

### Zero/Few-shot learning

Zero/Few-shot learning approaches can be run using the `zero_few_shot.py` script. It can be customized using several command line arguments such as:
- model_tag: tag or path of the model to use
- k_shot: number of examples to use
- few_shot_type: strategy to use to extract examples
- temperature/num_beams: temperature/number of beams generation hyper-parameter
- train_path: training dataset path to use to extract input examples
- test_path: test dataset path

*Details about the command to use the script will be added upon publication of the paper.*

### Chain-of-Though prompting

Chain-of-Though prompting can be run using the `zero_few_shot.py` script. It can be customized using several command line arguments such as:
- model_tag: tag or path of the model to use
- k_shot: number of examples to use
- scene_type: DREAM scene elaboration type to consider
- temperature/num_beams: temperature/number of beams generation hyper-parameter
- dream_train_path: training dataset path to use to extract input examples enriched with DREAM scene elaborations
- dream_test_path: test dataset path enriched with DREAM scene elaborations

*Details about the command to use the script will be added upon publication of the paper.*

### LLM fine-tuning

LLMs can be fine-tuned for the TER+Explanation task using FLUTE data using the `finetune.py` script.
It can be customized using several command line arguments such as:
- model_tag: tag or path of the model to fine-tune
- max_seq_length_src/max_seq_length_tgt: maximum number of input/output tokens
- learning_rate: learning rate for model fine-tuning
- num_train_epochs: number of epochs for model fine-tuning
- train_path/val_path: training/validation dataset path

*Details about the command to use the script will be added upon publication of the paper.*

### Testing

Fine-tuned LLMs can be tested using the `test.py` script.
It can be customized using several command line arguments such as:
- model_tag: tag or path of the model to test
- max_seq_length_src/max_seq_length_tgt: maximum number of input/output tokens
- output_folder: fine-tuned model path to test
- temperature/num_beams: temperature/number of beams generation hyper-parameter
- test_path: test dataset path

*Details about the command to use the script will be added upon publication of the paper.*

## Fine-tuned model checkpoints

Fine-tuned model checkpoints will be made available upon publication of the paper.

## Outputs

Model outputs will be made available upon verifying the dataset license.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
... to be added upon publication of the paper ...
```

## License

This project is licensed under the CC BY-NC-SA 4.0 License.
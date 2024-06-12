# T&T
Towards Certified Probabilistic Robustness
through Training and Testing

## Overview
This repository is dedicated to exploring and implementing models for probabilistic robustness in accuracy assessments. The focus is on developing methodologies that enhance the robustness of models against various types of input perturbations and uncertainties.

## Features
- **Ensemble Methods**: Utilization of ensemble techniques to aggregate predictions from multiple models to improve robustness.
- **Sampling Techniques**: Implementation of advanced sampling methods to evaluate the robustness under different scenarios.
- **Utility Scripts**: Includes various utility scripts for data handling and processing to facilitate robustness testing.

## Getting Started

### Prerequisites
Ensure you have Python installed on your machine. You can download Python from [python.org](https://www.python.org/downloads/).

### Installation
1. Clone the repository:
```
git clone https://github.com/soumission-anonyme/probabilistic-robustness-accuracy.git
```
2. Navigate to the repository directory:
```
cd probabilistic-robustness-accuracy
```
3. Install the required packages:
```
pip install -r requirements.txt
```

### To train the model using variance-minimizing training

In the ```experiments``` folder, we have ```e1.py``` is for MNIST, ```e2.py``` is for SVHN, and so on. Simply run the following code 

```
python experiments/e1.py
```

Also, we can toggle the step functions to select whether the inference (testing) method is with certification or not.

To view result, we need to use tensorboard as follows.
```
tensorboard --logdir=runs
```

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Decision Tree++

### A C++ re-implementation of [Avinash Kak's DecisionTree Module](https://pypi.org/project/DecisionTree/3.4.3/). 

This module allows users to construct a Decision Tree from training data containing either symbolic or numeric features, and peform classification tasks. A decision tree is a flowchart-like model utilizing nodes, where each internal node represents a "test" on an attribute designing to maximally-reduce entropy in a decision or classification task.​

## Practical Usage

Imagine you run a small investment company where traders make buy/sell decisions based on five specific criteria. One day, all your traders quit, however you have a record of all your traders' past decisions based on these criteria. Thankfully, our decision tree module can save your business!

By pooling past buy/sell data in a training file, you may build a decision tree to classify any future investment decisions. According to Avinash Kak, this approach would outperform any of your former individual traders, as the computer's buy/sell decisions would leverage the collective knowledge of all past traders.

![image](https://github.com/user-attachments/assets/59ce82b5-9c86-4422-ad6e-32a96f239036)

## Features
- Create a Decision Tree from either symbolic or numeric training data
- Produce classification results and output to the user
- Support INTEROPERABILITY: directly call our C++ code from Python programs
- Exceed the runtime performance of the original Python version
- Support the INTROSPECTION feature: explain the classification decisions made at the different nodes of the decision tree
- Quick ramp-up time via documentation and tests

## Requirements and Usage:
**Please consult our [UserManual.md](https://github.com/galaxybomb23/Decision-Tree-Plus-Plus/blob/main/UserManual.md)**.

## System Architecture

![image](https://github.com/user-attachments/assets/f4415800-379b-4843-8665-99e5b1f6b984)
![image](https://github.com/user-attachments/assets/00378c6c-719a-45cc-801e-904426ac21f5)


## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). You are free to use, modify, and distribute this code under the terms of the GPLv3.


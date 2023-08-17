# 🛰️ ORBIT Agent

_Optimized Risk Based Intelligence Trading_

***FOSS Financial Machine Learning Agent - Stock Classification | Stock Price Prediction***

## Contents

- [ORBIT Agent](#orbit-agent)
  - [Contents](#contents)
  - [Quick Start](#quick-start)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Training](#training)
      - [Classification](#classification)
      - [Price Prediction](#price-prediction)

## Quick Start

### Requirements
  [ ] Anaconda

### Installation
1. Clone the repo
  ```sh
    git clone https://github.com/newagemob/orbit-agent.git
  ```
2. Install Anaconda
  (Linux)
  ```sh
    https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
  ```
  (MacOS)
  ```sh
    brew install --cask anaconda
  ```
  (Windows)
  ```sh
    https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe
  ```
3. Activate the conda environment
  ```sh
    cd orbit-agent
  ```
  ```sh
    conda env create -f environment.yml
  ```
  ```sh
    conda activate orbit
  ```

### Training

#### Classification

To train a financial analysis model for classification, run the following command:
  ```sh
    python train.py --model classification
  ```

Once the model is trained, you can run the following command to test the model:
  ```sh
    python train.py --model classification --test
  ```

#### Price Prediction

To train a price prediction model (predicting the next day's closing price), run the following command:
  ```sh
    python train.py --model closing_price_prediction
  ```

Once the model is trained, you can run the following command to test the model:
  ```sh
    python train.py --model closing_price_prediction --test
  ```

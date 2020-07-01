# Pricing EV Charging Service with Demand Charge
This code and data are the companion to the paper *Pricing EV Charging Service with Demand Charge* which will appear in PSCC 2020.

## Getting Started

### Setup

#### 1. Clone the Repo
Clone this repository onto your local machine. 

#### 2. Install Dependencies
We recommend using a virtual environment to manage dependencies, you can create one by
 running 
 
`python -m venv venv`

Activate the virtual environment by running

`source venv/bin/activate`

You can then install all dependencies by running

`pip install -r requirments.txt`

#### 3. You are ready to go. 

### Running Experiments
#### Offline Pricing and Scheduling
To calculate prices and offline schedules as described in the paper, run

`python pricing_experiment`

*Note that this experiment can take a while to run. If you like, you can shorten the
 number of months to consider to speed it up. See line 115.
 
The code for actually calculating prices is found in `pricing_rule`.
 #### Online Scheduling
 To preform scheduling with only online information, run
 
 `python online_cost_minimization`

#### Plotting and Evaluation
All plotting and evaluation are contained in the Jupyter Notebook `evaluation.ipynb`.

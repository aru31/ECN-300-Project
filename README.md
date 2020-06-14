## ECN-300-Project

Create linguistic summary of different data sets.

##### To get the summary of any dataset:
```
git clone https://github.com/aru31/ECN-300-Project
```

Install the pip dependencies and enter the code directory
```
pip3 install -r requirements.txt
cd LinguisticSummary
```

For Iris dataset
```
python3 run_summary.py iris
```

For Wheat Seed dataset
```
python3 run_summary.py wheat_seed
```

To get summary of any other dataset, add a function in `run_summary.py`

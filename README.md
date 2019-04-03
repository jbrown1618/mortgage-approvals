# Mortgage Approvals

## Setup

```bash
git clone git@github.com:jbrown1618/mortgage-approvals.git
cd mortgage-approvals

virtualenv $(which python3) venv
source venv/bin/activate
pip install -r requirements.txt
echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc # bug with matplotlib on osx
deactivate
```

## Generating Images and Data

```bash
cd mortgage-approvals

source venv/bin/activate
python main.py
deactivate
```
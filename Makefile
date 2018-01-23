history:
	venv/bin/python main.py history

create_dataset:
	venv/bin/python main.py create_dataset

all: history create_dataset

init:
	venv/bin/pip install -r requirements.txt

history:
	venv/bin/python main.py history

history2:
	venv/bin/python main.py history2

train:
	venv/bin/python main.py train

create_dataset:
	venv/bin/python main.py create_dataset

predict: history2 train

init:
	venv/bin/pip install -r requirements.txt

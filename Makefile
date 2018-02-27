spot:
	venv/bin/python main.py spot

history:
	venv/bin/python main.py history

train:
	venv/bin/python main.py train

create_dataset:
	venv/bin/python main.py create_dataset

predict: history train

init:
	venv/bin/pip install -r requirements.txt --upgrade

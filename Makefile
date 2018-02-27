spot:
	venv/bin/python main.py spot

init:
	venv/bin/pip install -r requirements.txt --upgrade
	venv/bin/python main.py init

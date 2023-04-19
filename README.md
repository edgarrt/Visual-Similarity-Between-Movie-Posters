# Final Project: Visual Similarity Between Movie Posters


## Getting started with development

## Prereqs:
- Python 3.10+

## Check if you have Python 3.10 installed
`python3.10 --version`

If you get
`python3.10: command not found`

try:
`python3 --version`

If you get 
`Python 3.X.Y` where `X` is at least 10 then update the last line in the `Pipfile` to your output and continue to next section

ex:
```
main@LAPTOP:~/cosc525_final_project$ python3 --version
Python 3.10.6

# Pipfile
[requires]
python_version = "3.10"

```

Follow: [Installing Python 3.X](https://www.codingforentrepreneurs.com/blog/install-django-on-mac-or-linux/), but install the Python 3.10.6 binary

Should be this page - https://www.python.org/downloads/release/python-3106/




## Python Virtual Env
Create the virtual environment & install the requirements:
```
# this installs numpy automatically
pipenv install --dev

```
Enter the virtual environment:
```
pipenv shell
```



## Launch Jupyter notebook environment (Mostly for local development):
```
# This launches the notebook env & should open your browser
jupyter notebook
```


References:
- https://www.kaggle.com/code/mishki/resnet-keras-code-from-scratch-train-on-gpu
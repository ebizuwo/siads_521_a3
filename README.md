# Prerequisites
1. Must have python3 version installed
2. Must have git installed... assuming this is done already
3. Preferably running a machine with linux shell (installation file is a shell script)

# Installation
### Option 1
1. Clone repository on your local machine
2. Run the siads521_a2_install.sh (may require modifying the permissions of the script)
```shell
$ chmod 777 siads521_a3_install.sh
$ ./siads521_a2_install.sh
```

### Option 2
1. Clone repository
2. Install python 3
3. Setup virtual environment
```shell 
$ python3 -m venv ass3
```
Then activate the virtual environment 
```shell
$ source ass3/bin/activate
```
4. Install requirements
```shell
$ pip install -r requirements.txt
```
5. Run Jupyter Notebook server
```shell
$ jupyter notebook
```
6. Done


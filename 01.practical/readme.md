# Best practice to setting up environment 

## 1. Creating virtual environment
### Anaconda/ miniconda 
The easiest way to set up the environment is to use Anaconda or Miniconda.

Anaconda comes with everything we need (and much more). Miniconda is a smaller version of Anaconda that contains only Python.

**How to do**   

In your terminal, run this command to create the environment
```
conda create -n env_name python=3.9
```
Activate it:
```
conda activate env_name
```
Installing libraries inside my virtual environment
```
conda install numpy pandas scikit-learn seaborn jupyter
```

## 2. Use Cloud service 
### AWS
* AWS EC2 instance
### GCP 



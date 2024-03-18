# Learning Curve Fitting

Learning Curve Fitting Paper Experiments

## Running out experiments

First, we need to download the LCDB 
```
wget ...
```


Download our image:

```
singularity build learningcurvefitting.sif docker://taylanot/learningcurvefitting:latest
```

Then one can run the experiments with the image:
```
singularity exec learningcurvefitting.sif python3 main.py -F experiments
```


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
Above code will save the experiment in a unique file with all the run information. If you want to change configuration from the command line just do:
```
singularity exec learningcurvefitting.sif python3 main.py -F experiments with conf.x=y
```
where x and y are related parts that you want to change. For setting the seed following command will do.
```
singularity exec learningcurvefitting.sif python3 main.py -F experiments with seed=24
```


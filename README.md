### Setup 

Run ```process.R``` to convert the raw data from the SPSS file into an R dataframe (note: original SPSS file is not in this repo). This will also generate other useful files in the ```workspace``` folder.

##### Sleep Data 

To get Alspac sleep variables, run ```prepare.R``` in the ```sleep``` folder. This will create a dataframe for sleep data. The file ```learn.py``` can be used to learn a State Space Model using PyStan based on the timeseries sleep data. The ```program_miss.stan``` file describes the model (which can handle missing data). The file ```vb_learn.py``` is identical to ```learn.py``` but uses variational inference instead of MCMC. 

##### Social Behaviour Data 

To setup the social behaviour data, successively run ```prepare_1.py``` and ```prepare_2.R``` (sorry). The ```vae_learn.py``` script can be used to learn a VAE model on a single time point of the social behaviour timeseries. The ```train.py``` script can be used to train a Sequential VAE model on the entire timeseries. It can also be used to train the model on simulated data. Models are built in tensorflow and training is done using the tf.Estimator class. 

###### Other Notes 

The ```var()``` method in ```search.py``` can be used to search for variable names within the dataframe.
 

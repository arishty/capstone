# NYCDSA Capstone Project

In this project, we worked with a US travel recommendation startup, automating processes such as categorizing events in the US with supervised and unsupervised learning methods.


* To view the finalized documented notebook of our work, please refer to Event_Processing_Models_With_Thresholds_pickled.ipynb.

* To use our CLI tool, one can navigate to the LTD_model_deploy directory and run "python model_deploy.py -d [insert event description] -v [insert 0,1, or 2]" to output label predictions.
- Use -v 0 to only output the label predictions and corresponding probabilities
- Use -v 1 to output the label predictions and probabilities as well as the original event description enterred
- Use -v 2 to output all of the above, as well as the preprocessed set of words that the classifier evaluates

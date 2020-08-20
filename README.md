# Serum_ML

The script final_predictor_kbest.py performs a feature selection by Recursive Feature
Elimination with Cross-Validation, and utilizesthe selected features for training the
serum classifier. Run the script after cloning without any parameters.An output folder
will be created with the results. The classifier is trained on 54 samples and its 
performance is testes on 33 samples.The script utilizes the protein intensity matrix 
LFQ_intensity.txt and the experimental design file expdesign.txt from the data folder
to train and test the classifier.



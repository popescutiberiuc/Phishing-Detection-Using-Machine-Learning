This is a project that aims to make it easier for the users to detect phishing emails out of their Gmail inbox.
It uses a Logistic Regression model as well as a Random Forest model that then perform weighted voting in order to land on a decision: is it phishing or not phishing?
A broswer extension was later created for ease of use.
The performance of the model on real .eml files from the SpamAssassing dataset (400 samples with equal distribution) had an F1 score of 0.92 while the performance on the 80,000 total dataset from kaggle (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) had an F1 score of around 0.98. 

Future work:
There were some limitations I encountered that can be worked on. The training set is made up of English only emails which means that the project is not effective for non-english emails, furthure feature engineering can be conducted and a more reliable, better structured dataset should be used. The current project uses Gmail's HMTL structure to extract data from the inbox so the extension only works in the Gmail client atm. 

For current purposes, the application still works with good results.

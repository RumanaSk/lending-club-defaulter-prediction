### Project Overview

 Lending Club is a peer to peer lending company based in the United States, in which investors provide funds for potential borrowers and investors earn a profit depending on the risk they take (the borrowers credit score).
Lending Club provides the "bridge" between investors and borrowers. This data contains complete loan data for all loans issued for the first quarter of 2012, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Our goal here is to predict potential loan defaulters.
Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. The file is a matrix of about 188183 observations and 77 variables

Feature	Description
addr_state	The state provided by the borrower in the loan application
annual_inc	The self-reported annual income provided by the borrower during registration
annual_inc_joint	The combined self-reported annual income provided by the co-borrowers during registration
application_type	Indicates whether the loan is an individual application or a joint application with two co-borrowers
collection_recovery_fee	post charge off collection fee
collections_12_mths_ex_med	Number of collections in 12 months excluding medical collections
delinq_2yrs	The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
desc	Loan description provided by the borrower
dti	A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
dti_joint	A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income
earliest_cr_line	The month the borrower's earliest reported credit line was opened
emp_length	Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
emp_title	The job title supplied by the Borrower when applying for the loan.*
fico_range_high	The upper boundary range the borrower’s FICO at loan origination belongs to.
fico_range_low	The lower boundary range the borrower’s FICO at loan origination belongs to.
funded_amnt	The total amount committed to that loan at that point in time.
funded_amnt_inv	The total amount committed by investors for that loan at that point in time.
grade	LC assigned loan grade
home_ownership	The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
id	A unique LC assigned ID for the loan listing.
initial_list_status	The initial listing status of the loan. Possible values are – W, F
inq_last_6mths	The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
installment	The monthly payment owed by the borrower if the loan originates.
int_rate	Interest Rate on the loan
is_inc_v	Indicates if income was verified by LC, not verified, or if the income source was verified
issue_d	The month which the loan was funded
last_credit_pull_d	The most recent month LC pulled credit for this loan
last_fico_range_high	The upper boundary range the borrower’s last FICO pulled belongs to.
last_fico_range_low	The lower boundary range the borrower’s last FICO pulled belongs to.
last_pymnt_amnt	Last total payment amount received
last_pymnt_d	Last month payment was received
loan_amnt	The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
loan_status	Current status of the loan
member_id	A unique LC assigned Id for the borrower member.
mths_since_last_delinq	The number of months since the borrower's last delinquency.
mths_since_last_major_derog	Months since most recent 90-day or worse rating
mths_since_last_record	The number of months since the last public record.
next_pymnt_d	Next scheduled payment date
open_acc	The number of open credit lines in the borrower's credit file.
out_prncp	Remaining outstanding principal for total amount funded
out_prncp_inv	Remaining outstanding principal for portion of total amount funded by investors
policy_code	publicly available policycode=1 new products not publicly available policycode=2
pub_rec	Number of derogatory public records
purpose	A category provided by the borrower for the loan request.
pymnt_plan	Indicates if a payment plan has been put in place for the loan
recoveries	post charge off gross recovery
revol_bal	Total credit revolving balance
revol_util	Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
sub_grade	LC assigned loan subgrade
term	The number of payments on the loan. Values are in months and can be either 36 or 60.
title	The loan title provided by the borrower
total_acc	The total number of credit lines currently in the borrower's credit file
total_pymnt	Payments received to date for total amount funded
total_pymnt_inv	Payments received to date for portion of total amount funded by investors
total_rec_int	Interest received to date
total_rec_late_fee	Late fees received to date
total_rec_prncp	Principal received to date
url	URL for the LC page with listing data.
verified_status_joint	Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
zip_code	The first 3 numbers of the zip code provided by the borrower in the loan application.
open_acc_6m	Number of open trades in last 6 months
open_il_6m	Number of currently active installment trades
open_il_12m	Number of installment accounts opened in past 12 months
open_il_24m	Number of installment accounts opened in past 24 months
mths_since_rcnt_il	Months since most recent installment accounts opened
total_bal_il	Total current balance of all installment accounts
il_util	Ratio of total current balance to high credit/credit limit on all install acct
open_rv_12m	Number of revolving trades opened in past 12 months
open_rv_24m	Number of revolving trades opened in past 24 months
max_bal_bc	Maximum current balance owed on all revolving accounts
all_util	Balance to credit limit on all trades
total_rev_hi_lim	Total revolving high credit/credit limit
inq_fi	Number of personal finance inquiries
total_cu_tl	Number of finance trades
inq_last_12m	Number of credit inquiries in past 12 months
acc_now_delinq	The number of accounts on which the borrower is now delinquent.
tot_coll_amt	Total collection amounts ever owed
tot_cur_bal	Total current balance of all accounts




### Learnings from the project

 Random Forest, XG Boost Implementation


### Approach taken to solve the problem

 ## Load the data
### Instructions
•	Load dataset using pandas read_csv method in variable df and give file path as filepath_or_buffer=path, compression='zip'and low_memory = False
•	Store all the features(independent values) in a variable called X
•	Store the target variable (loan_status) in a variable called y
•	Split the dataframe into X_train,X_test,y_train,y_test using train_test_split() function. Use test_size = 0.25 and random_state = 4
## Data cleaning
### Instructions
•	Find the sum of null values for each column and store it in a variable col
•	Find the features with more than 25% missing data and add it to a variable col_drop
•	Find columns which contains only one unique value using.nunique() and append it to the variable col_drop
•	Drop the features stored in the variable col_drop from X_trainand X_test

##Multiclass to binary class
###Observation
Below we can see that in the target variable loan_status there are six classes. We want to convert these six classes to two classes. So we put Fully Paid and Current as one class as it is very unlikely that these will be defaulters and the rest to the other class.
INPUT
df.loan_status.value_counts()
 
##Missing values and Encoding categorical variables
###Instructions
•	The numerical variables are stored in num and categorical variables are stored in cat
•	Fill the missing values in X_train and X_test with mean for numerical variable and mode for categorical variables.
•	Label Encode categorical variables in X_train and X_test.
##Random Forest
###Instructions
•	Instantiate RandomForestClassifier to a variable rf with random_state =42,max_depth=2andmin_samples_leaf=5000.
•	Fit the model on X_train and y_train.
•	Store the f1_score in variable f1, precision_score in variable precision, recall_score in variable recall and roc_auc_scorein roc_auc. **Note: Write rocaucscore(ytest, ypred)
•	Print the confusion_matrix and classification_report.
•	Predict the probability for the X_test == 1 and store the result in y_pred_proba.
•	Use metrics.roc_curve to calculate the fpr and tpr and store the result in variables fpr, tpr,.
•	Calculate the auc score of y_test and y_pred_proba and store it in variable called auc.
•	Plot auc curve of 'auc' using the line plt.plot(fpr,tpr,label="Random Forest model, auc="+str(auc)).

##XG Boost
###Instructions
•	Instantiate XGBoostClassifier and store it to a variable xgb with parameter learning_rate = 0.0001
•	Fit the model on X_train and y_train.
•	Store the values predicted by model for X_test in a variable y_pred.
•	Store the f1_score in variable f1, precision_score in variable precision, recall_score in variable recall and roc_auc_scorein roc_auc.
•	Print the confusion_matrix and classification_report.
•	Predict the probability for the X_test == 1 and store the result in y_pred_proba.
•	Use metrics.roc_curve to calculate the fpr and tpr and store the result in variables fpr, tpr,_.
•	Calculate the auc score of y_test and y_pred_proba and store it in variable called auc.
•	Plot auc curve of 'auc' using the line plt.plot(fpr,tpr,label="XGBoost model, auc="+str(auc)).






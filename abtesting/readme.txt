1. Run ./runHive.sh on kandula editing the below parameters
    a. startDate = Start Date of AB test
    b. endDate = Start Date of AB test 
    c. fileName = file name on s3 to save results of query. Directory="user/hive/warehouse/DsAbTestingFinal/fileName/" 

2. Run ./runAnalysis.sh script locally editing the below parameters
	***All libraries needed for this script is in requirements.txt***
	***For pymmsql http://gree2.github.io/python/setup/2017/04/19/python-instal-pymssql-on-mac***
    a. First variable: Name of file from 1.c above
    b. Second variable: Comma seperated Client names for which to run the analysis for
    c. Third variable: Comma seperated dates for which to include the results

Script above produces two results. One for ctr/rpm analysis and one for cr/roas analysis. 

3. Edit all .csv files
	a. Highlight all numbers and change decimal points to 2 values;
	   for CTR value, change to 4 decimal points for better comparison.
	b. Add "%" for the following headers
		- ctr
		- cr
		- ctrLift
		- rpmLift
		- roasLift
		- crLift
		- aovLift
		- cr
	c. Add "%(+/-)" for the all the headers with "CI" (confidence intervals)
	d. Add "$" for all the following headers:
		- spend
		- rpm
		- revenue
		- roas
		- aov

Common errors and comments:
1. If permission error to get algoUI then connect to HL vpn and then try again
2. Edit code for something besides Global and big retailer breakdown


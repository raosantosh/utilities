import sys
import lxml
import boto3
import math
import pymssql
import bs4 as bs
import pandas as pd
import numpy as np
from scipy.stats import norm

pd.set_option('display.float_format', lambda x: '%.2f' % x)

'''
This script parses algo UI xml file from DB7
'''
'''
python CtrAbTest/abAnalysisTemplate.py testingHisham 5313 QS
'''
print ("accessing DB7 for xml parsing")
conn = pymssql.connect(server="retail-db.hooklogic.com",
                        user="DataScienceReader",
                        password="CFCzw4YAJRhyAchO8nkj")
query = '''select segment from leadgeneration.dbo.clientpathconfigurationfiles
            with(nolock) where configType = 'GlobalAlgo' '''
cursor = conn.cursor()
cursor.execute(query)
content = cursor.fetchall()
conn.close()

print ("parsing algo ui xml file")
soup = bs.BeautifulSoup(content[0][0],'xml')
allAlgos = soup.find_all("AlgorithmConfiguration")

algoDict = dict()
counter=1
for algo in allAlgos:
    if algo.find_all('Clients')[0].text!='':
        currentAlgo = dict()
        currentAlgo['algoId'] = algo.find_all('ID')[0].text
        currentAlgo['algoName'] = algo.find_all('Name')[0].text
        currentAlgo['cid'] = algo.find_all('Clients')[0].find_all('int')[0].text
        currentAlgo['status'] = algo.find_all('Status')[0].text
        currentAlgo['placementType'] = '/'.join([i for i in algo.find_all("PlacementTypes")[0].text.split('\n') if len(i)>0])
        currentAlgo['startDate'] = algo.find_all('TestStartDate')[0].text.split('T')[0]
        currentAlgo['endDate'] = algo.find_all('TestEndDate')[0].text.split('T')[0]
        currentAlgo['traffic'] = algo.find_all('UsagePercentage')[0].text
        currentAlgo['useKWMCandiate'] = 'false' if algo.find('UseCandidateKeywordsModel') is None else algo.find('UseCandidateKeywordsModel').text
        algoDict[counter] = currentAlgo
        counter = counter+1

print ("Building algo UI dataframe")
algoDf = pd.DataFrame.from_dict(algoDict,orient='index')
algoDf['algoId'] = [int(i) for i in algoDf.algoId]
algoDf = algoDf[algoDf['algoId']!=-1]
algoDf=algoDf[algoDf['status']=='Active']
#Get data from s3
access_key_id = "AKIAJ4CC4RABZGCAPFNA"
access_secret_key = "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"
bucket_name = 'datascience-shared'
prefix="user/hive/warehouse/DsAbTestingFinal/" + sys.argv[1] + "/"
#prefix="user/hive/warehouse/DsAbTestingFinal/" + "abtest_9_7" + "/"

s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=access_secret_key
    )
keyList = s3.list_objects(Bucket=bucket_name, Prefix=prefix)["Contents"]

print("Path to prefix " + prefix)
df = pd.DataFrame()
for keyObj in keyList:
    print(keyObj)
    if keyObj["Size"]!=0:
        key = keyObj["Key"]
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        text_file = open("/tmp/temp_output.txt", "w")
        text_file.write(obj['Body'].read().decode('utf-8'))
        text_file.close()
        df = pd.concat([df,pd.read_csv("/tmp/temp_output.txt", "\t",header=None)], axis=0)
'''
df = pd.DataFrame()
for i in range(0,5):
    df = df.append(pd.read_csv("~/Desktop/CtrAbTest/00000" + str(i) + "_0", header=None, sep='\t'))
'''
df.columns = ["cid","algoId","dt","impressions","click","spend","conv","revenue"]
df = df.merge(algoDf, on="algoId")

#group1 = "A-"+sys.argv[2]
group1 = "A-BaseModel"
print (group1)
#group2 = "B-"+sys.argv[3]
group2 = "B-CandidateModel"
print (group2)
'''
group1 = "5313"
group2 = "QS"
'''

clients = sys.argv[2].split(",")
dates = sys.argv[3].split(",")

print(clients)
print(dates)

'''
df["name"] = np.where(df['algoName'].str.contains(group2.split("-")[1]), group2,
                        (np.where(df['algoName'].str.contains("5311"), "5311",
                            (np.where(df['algoName'].str.contains(group1.split("-")[1]), group1,
                                (np.where(df['algoName'].str.contains("N1"), "N1","Others")
                                        )
                                    )
                                  )
                                )
                            )
                        )
'''
df['name'] = df['useKWMCandiate'].map({'true': group2, 'false': group1})
df = df[(df['algoName'].str.find("N1") == -1) & (df['algoName'].str.find("NewRecoFix") != -1)]
df = df[df["cid_x"].isin(clients)]
df = df[df["dt"].isin(dates)]
df["traffic"] = [float(i) for i in df["traffic"]]
df = df.replace("\\N", 0)
df['conv'] = [float(i) for i in df.conv]
df['revenue'] = [float(i) for i in df.revenue]

#df = df[(df["name"]==group2)|(df["name"]==group1)]
#df = df[(df["cid_x"]!="Other") & (df["cid_x"]!='Peapod') & (df['algoName'].str.find("N1") == -1)]
'''
Conversions to make for revenue and spend
Argos: Pounds to USD
Asda: Pounds to USD
Bol.com: euro to USD
FNAC: euro to USD
NoteBooksBillinger: euro to USD
'''
for pub in ["Argos", "Asda"]:
    df.loc[df.cid_x==pub,"spend"] = df[df["cid_x"]==pub]["spend"]*1.29
    df.loc[df.cid_x==pub,"revenue"] = df[df["cid_x"]==pub]["revenue"]*1.29

for pub in ["FNAC", "NoteBooksBillinger"]:
    df.loc[df.cid_x==pub,"spend"] = df[df["cid_x"]==pub]["spend"]*1.18
    df.loc[df.cid_x==pub,"revenue"] = df[df["cid_x"]==pub]["revenue"]*1.18


#scale according to algoUI
#df = df.groupby(["cid_x","dt","name","traffic"], as_index=False)[["impressions","click","spend","conv","revenue"]].sum()
df = df.groupby(["cid_x","dt","name"], as_index=False)[["impressions","click","spend","conv","revenue","traffic"]].sum()
df["impressions_real"] = df["impressions"]
df["click_real"] = df["click"]
df["spend_real"] = df["spend"]
df["conv_real"] = df["conv"]
df["revenue_real"] = df["revenue"]

df["impressions"] = df["impressions"]*(100/df["traffic"])
df["click"] = df["click"]*(100/df["traffic"])
df["spend"] = df["spend"]*(100/df["traffic"])
df["conv"] = df["conv"]*(100/df["traffic"])
df["revenue"] = df["revenue"]*(100/df["traffic"])



#for conversion test
def bootstrapInt(retailerData, feature, num_of_sample, group1, group2):
    diffSamp = []
    print(retailerData.columns)
    g1Val = retailerData[retailerData["name"]==group1].sort_values(by='dt')[feature]
    g2Val = retailerData[retailerData["name"]==group2].sort_values(by='dt')[feature]
    g1Val.index = np.arange(1, len(g1Val) + 1)
    g2Val.index = np.arange(1, len(g2Val) + 1)
    diffVal =  g1Val - g2Val
    for i in range(0, num_of_sample):
        diffSamp.append(np.mean(np.random.choice(diffVal,len(diffVal), replace=True)))
    return np.percentile(diffSamp, 95)

convReportFinal = pd.DataFrame()
for grp in clients + ["Total"]:
    print('Processing for ' + grp)
    if grp=="Total":
        convDf = df.groupby(["name","dt"], as_index=False)[["revenue","spend","conv","click","impressions","revenue_real","spend_real","conv_real","click_real","impressions_real"]].sum()
        convDf["retailer"] = "Total"
    else:
        convDf = df[df["cid_x"] == grp].groupby(["name","dt"], as_index=False)[["revenue","spend","conv","click","impressions","revenue_real","spend_real","conv_real","click_real","impressions_real"]].sum()
        convDf["retailer"] = grp
    convDf["roas"] = convDf["revenue"]/convDf["spend"]
    convDf["cr"] = convDf["conv"]/convDf["click"]
    convDf["aov"] = convDf["revenue"]/convDf["conv"]
    convReport = convDf.groupby(["name","retailer"], as_index=False)[["revenue","spend","conv","click","impressions","revenue_real","spend_real","conv_real","click_real","impressions_real"]].sum()
    convReport = convReport.pivot(index="retailer",columns="name")[["impressions","click","spend","conv","revenue","revenue_real","spend_real","conv_real","click_real","impressions_real"]]

    for grp in [group1,group2]:
        convReport["roas",grp] = (convReport['revenue']/convReport['spend'])[grp]
    for grp in [group1,group2]:
        convReport["cr",grp] = (convReport['conv']/convReport['click'])[grp]
    for grp in [group1,group2]:
        convReport["aov",grp] = (convReport['revenue']/convReport['conv'])[grp]

    print ("calculating intervals for cr and roas")
    for feature in ["roas", "cr", "aov"]:
        convReport[feature+"Lift"] = (convReport[feature][group2]-convReport[feature][group1])/convReport[feature][group1]
        highCI = bootstrapInt(convDf, feature, num_of_sample=1000, group1=group2, group2=group1)
        convReport[feature+"CI"] = (highCI/convReport[feature][group1])-convReport[feature+"Lift"]
        convReport[feature+"Lift"] = convReport[feature+"Lift"]*100
        convReport[feature+"CI"] = convReport[feature+"CI"]*100
    convReport["cr"] = convReport["cr"]*100
    convReportFinal = pd.concat([convReportFinal,convReport], axis=0)

convReportFinal.to_csv("~/Desktop/convReport.csv")
#for ctr and rpm
#df = df.groupby(["cid_x","dt","name"], as_index=False)[["impressions","click","spend"]].sum()
df = df.groupby(["cid_x","dt","name"], as_index=False)[["impressions","click","spend","impressions_real","click_real","spend_real"]].sum()

#dfReport = df.groupby(["cid_x","name"], as_index=False)[["impressions","click","spend"]].sum()
dfReport = df.groupby(["cid_x","name"], as_index=False)[["impressions","click","spend","impressions_real","click_real","spend_real"]].sum()

#dfReport = dfReport.pivot(index="cid_x", columns="name")[["impressions","click","spend"]]
dfReport = dfReport.pivot(index="cid_x", columns="name")[["impressions","click","spend", "impressions_real","click_real","spend_real"]]
dfReport.loc["Total"]=dfReport.sum()
dfReport["ctr",group1] = (dfReport['click']/dfReport['impressions'])[group1]
dfReport["ctr",group2] = (dfReport['click']/dfReport['impressions'])[group2]
dfReport["rpm", group1] = (dfReport['spend']/(dfReport['impressions']/1000))[group1]
dfReport["rpm", group2] = (dfReport['spend']/(dfReport['impressions']/1000))[group2]

# Imp, click, spend lift
dfReport["impLift"] = (dfReport["impressions"][group2]-dfReport["impressions"][group1])/(dfReport["impressions"][group1]) * 100
dfReport["clickLift"] = (dfReport["click"][group2]-dfReport["click"][group1])/(dfReport["click"][group1]) * 100
dfReport["spendLift"] = (dfReport["spend"][group2]-dfReport["spend"][group1])/(dfReport["spend"][group1]) * 100

# ctr calculations
dfReport["ctrLift"] = (dfReport["ctr"][group2]-dfReport["ctr"][group1])/(dfReport["ctr"][group1])

def ctrInt(n1, p1, n2, p2):
    SE = math.sqrt(   (p1*(1-p1)/n1)+(p2*(1-p2)/n2)   )
    z_star = norm.ppf(.95)
    ME = z_star*SE
    return ((p1-p2)/p2)-((p1-p2-ME)/p2)

ctrConfInt = []
for retailer in dfReport.index:
    n1 = dfReport.loc[retailer]["impressions"][group1]
    n2 = dfReport.loc[retailer]["impressions"][group2]
    p1 = dfReport.loc[retailer]["ctr"][group1]
    p2 = dfReport.loc[retailer]["ctr"][group2]
    ctrConfInt.append(ctrInt(n1, p1, n2, p2))

dfReport["ctrLift"] = dfReport["ctrLift"]*100
dfReport["ctr"] = dfReport["ctr"]*100
dfReport["ctrPlusMinusPercentage"] = [i*100 for i in ctrConfInt]

# rpm calculations
dfReport["rpmLift"] = (dfReport["rpm"][group2]-dfReport["rpm"][group1])/(dfReport["rpm"][group1])
dfReport["impLift"] = (dfReport["impressions"][group2]-dfReport["impressions"][group1])/(dfReport["impressions"][group1])

dfRpm = df
dfRpm["rpm"] = dfRpm["spend"]/(dfRpm["impressions"]/1000)

print ("calculating intervals for ctr and rpm")
plusMinusInt = []
plusMinusImpInt = []
for retailer in dfReport.index:
    if retailer=="Total":
        #retailerData = dfRpm.groupby(["dt","name"],as_index=False)["impressions","click","spend"].sum()
        retailerData = dfRpm.groupby(["dt","name"],as_index=False)["impressions","click","spend","impressions_real","click_real","spend_real"].sum()
        retailerData["rpm"] = retailerData["spend"]/(retailerData["impressions"]/1000)
    else:
        retailerData = dfRpm[dfRpm["cid_x"]==retailer]
    print(retailerData)
    rpm95 = bootstrapInt(retailerData, "rpm", num_of_sample=1000, group1=group2, group2=group1)
    imp95 = bootstrapInt(retailerData, "impressions", num_of_sample=1000, group1=group2, group2=group1)
    plusMinusInt.append(rpm95/dfReport.loc[retailer]["rpm"][group1])
    plusMinusImpInt.append((imp95 * (len(retailerData) / 2.0))/dfReport.loc[retailer]["impressions"][group1])

dfReport["rpmPlusMinusPercentage"]=(plusMinusInt - dfReport["rpmLift"])*100
dfReport["rpmLift"] = dfReport["rpmLift"]*100
dfReport["impPlusMinusPercentage"]=(plusMinusImpInt - dfReport["impLift"])*100
dfReport["impLift"] = dfReport["impLift"]*100

dfReport = dfReport.loc[["Target", "Peapod","Kohls", "Staples","Toys R Us ATG", "Macys", "Kmart", "Sears", "Costco", "Jet", "Total"],:]

dfReport.to_csv("~/Desktop/ctrRpmReport.csv")

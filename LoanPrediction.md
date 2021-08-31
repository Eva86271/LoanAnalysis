```python
import pandas as pd
import numpy as np
import csv
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2 
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score,confusion_matrix
```


```python
data1=pd.read_csv('C:/Users/subha/OneDrive/Desktop/train_ctrUa4K.csv',header=0)
```


```python
print(data1.head())
```

        Loan_ID Gender Married Dependents     Education Self_Employed  \
    0  LP001002   Male      No          0      Graduate            No   
    1  LP001003   Male     Yes          1      Graduate            No   
    2  LP001005   Male     Yes          0      Graduate           Yes   
    3  LP001006   Male     Yes          0  Not Graduate            No   
    4  LP001008   Male      No          0      Graduate            No   
    
       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \
    0             5849                0.0         NaN             360.0   
    1             4583             1508.0       128.0             360.0   
    2             3000                0.0        66.0             360.0   
    3             2583             2358.0       120.0             360.0   
    4             6000                0.0       141.0             360.0   
    
       Credit_History Property_Area Loan_Status  
    0             1.0         Urban           Y  
    1             1.0         Rural           N  
    2             1.0         Urban           Y  
    3             1.0         Urban           Y  
    4             1.0         Urban           Y  
    


```python
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 614 entries, 0 to 613
    Data columns (total 13 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Loan_ID            614 non-null    object 
     1   Gender             601 non-null    object 
     2   Married            611 non-null    object 
     3   Dependents         599 non-null    object 
     4   Education          614 non-null    object 
     5   Self_Employed      582 non-null    object 
     6   ApplicantIncome    614 non-null    int64  
     7   CoapplicantIncome  614 non-null    float64
     8   LoanAmount         592 non-null    float64
     9   Loan_Amount_Term   600 non-null    float64
     10  Credit_History     564 non-null    float64
     11  Property_Area      614 non-null    object 
     12  Loan_Status        614 non-null    object 
    dtypes: float64(4), int64(1), object(8)
    memory usage: 62.5+ KB
    


```python
# Handling missing values 

data1.loc[:,'Loan_ID']=data1.loc[:,'Loan_ID'].str.strip()
data1.loc[:,'Gender']=data1.loc[:,'Gender'].str.strip()
data1.loc[:,'Dependents']=data1.loc[:,'Dependents'].str.strip()
data1.loc[:,'Married']=data1.loc[:,'Married'].str.strip()
data1.loc[:,'Self_Employed']=data1.loc[:,'Self_Employed'].str.strip()
data1.loc[:,'Property_Area']=data1.loc[:,'Property_Area'].str.strip()
data1.loc[:,'Loan_Status']=data1.loc[:,'Loan_Status'].str.strip()

```


```python
data1=data1.set_index('Loan_ID')
```


```python
for i in data1.columns:
    print(data1.loc[:,i].unique())    
```

    ['Male' 'Female' nan]
    ['No' 'Yes' nan]
    ['0' '1' '2' '3+' nan]
    ['Graduate' 'Not Graduate']
    ['No' 'Yes' nan]
    [ 5849  4583  3000  2583  6000  5417  2333  3036  4006 12841  3200  2500
      3073  1853  1299  4950  3596  3510  4887  2600  7660  5955  3365  3717
      9560  2799  4226  1442  3750  4166  3167  4692  3500 12500  2275  1828
      3667  3748  3600  1800  2400  3941  4695  3410  5649  5821  2645  4000
      1928  3086  4230  4616 11500  2708  2132  3366  8080  3357  3029  2609
      4945  5726 10750  7100  4300  3208  1875  4755  5266  1000  3333  3846
      2395  1378  3988  2366  8566  5695  2958  6250  3273  4133  3620  6782
      2484  1977  4188  1759  4288  4843 13650  4652  3816  3052 11417  7333
      3800  2071  5316  2929  3572  7451  5050 14583  2214  5568 10408  5667
      2137  2957  3692 23803  3865 10513  6080 20166  2014  2718  3459  4895
      3316 14999  4200  5042  6950  2698 11757  2330 14866  1538 10000  4860
      6277  2577  9166  2281  3254 39999  9538  2980  1863  7933  3089  4167
      9323  3707  2439  2237  8000  1820 51763  3522  5708  4344  3497  2045
      5516  6400  1916  4600 33846  3625 39147  2178  2383   674  9328  4885
     12000  6033  3858  4191  3125  8333  1907  3416 11000  4923  3992  3917
      4408  3244  3975  2479  3418  3430  7787  5703  3173  3850   150  3727
      5000  4283  2221  4009  2971  7578  3250  4735  4758  2491  3716  3189
      3155  5500  5746  3463  3812  3315  5819  2510  2965  3406  6050  9703
      6608  2882  1809  1668  3427  2661 16250  3083  6045  5250 14683  4931
      6083  2060  3481  7200  5166  4095  4708  4333  2876  3237 11146  2833
      2620  3900  2750  3993  3103  4100  4053  3927  2301  1811 20667  3158
      3704  4124  9508  3075  4400  3153  4416  6875  4666  2875  1625  2000
      3762 20233  7667  2917  2927  2507  2473  3399  2058  3541  4342  3601
      3166 15000  8666  4917  5818  4384  2935 63337  9833  5503  1830  4160
      2647  2378  4554  2499  3523  6333  2625  9083  8750  2666  2423  3813
      3875  5167  4723  4750  3013  6822  6216  5124  6325 19730 15759  5185
      3062  2764  4817  4310  3069  5391  5941  7167  4566  2346  3010  5488
      9167  9504  1993  3100  3276  3180  3033  3902  1500  2889  2755  1963
      7441  4547  2167  2213  8300 81000  3867  6256  6096  2253  2149  2995
      1600  1025  3246  5829  2720  7250 14880  4606  5935  2920  2717  8624
      6500 12876  2425 10047  1926 10416  7142  3660  7901  4707 37719  3466
      3539  3340  2769  2309  1958  3948  2483  7085  3859  4301  3708  4354
      8334  2083  7740  3015  5191  2947 16692   210  3450  2653  4691  5532
     16525  6700  2873 16667  4350  3095 10833  3547 18333  2435  2699  5333
      3691 17263  3597  3326  4625  2895  6283   645  3159  4865  4050  3814
     20833  3583 13262  3598  6065  3283  2130  5815  2031  3074  4683  3400
      2192  5677  7948  4680 17500  3775  5285  2679  6783  4281  3588 11250
     18165  2550  6133  3617  6417  4608  2138  3652  2239  3017  2768  3358
      2526  2785  6633  2492  2454  3593  5468  2667 10139  3887  4180  3675
     19484  5923  5800  8799  4467  3417  5116 16666  6125  6406  3087  3229
      1782  3182  6540  1836  1880  2787  2297  2165  2726  9357 16120  3833
      6383  2987  9963  5780   416  2894  3676  3987  3232  2900  4106  8072
      7583]
    [0.00000000e+00 1.50800000e+03 2.35800000e+03 4.19600000e+03
     1.51600000e+03 2.50400000e+03 1.52600000e+03 1.09680000e+04
     7.00000000e+02 1.84000000e+03 8.10600000e+03 2.84000000e+03
     1.08600000e+03 3.50000000e+03 5.62500000e+03 1.91100000e+03
     1.91700000e+03 2.92500000e+03 2.25300000e+03 1.04000000e+03
     2.08300000e+03 3.36900000e+03 1.66700000e+03 3.00000000e+03
     2.06700000e+03 1.33000000e+03 1.45900000e+03 7.21000000e+03
     1.66800000e+03 1.21300000e+03 2.33600000e+03 3.44000000e+03
     2.27500000e+03 1.64400000e+03 1.16700000e+03 1.59100000e+03
     2.20000000e+03 2.25000000e+03 2.85900000e+03 3.79600000e+03
     3.44900000e+03 4.59500000e+03 2.25400000e+03 3.06600000e+03
     1.87500000e+03 1.77400000e+03 4.75000000e+03 3.02200000e+03
     4.00000000e+03 2.16600000e+03 1.88100000e+03 2.53100000e+03
     2.00000000e+03 2.11800000e+03 4.16700000e+03 2.90000000e+03
     5.65400000e+03 1.82000000e+03 2.30200000e+03 9.97000000e+02
     3.54100000e+03 3.26300000e+03 3.80600000e+03 3.58300000e+03
     7.54000000e+02 1.03000000e+03 1.12600000e+03 3.60000000e+03
     2.33300000e+03 4.11400000e+03 2.28300000e+03 1.39800000e+03
     2.14200000e+03 2.66700000e+03 8.98000000e+03 2.01400000e+03
     1.64000000e+03 3.85000000e+03 2.56900000e+03 1.92900000e+03
     7.75000000e+03 1.43000000e+03 2.03400000e+03 4.48600000e+03
     1.42500000e+03 1.66600000e+03 8.30000000e+02 3.75000000e+03
     1.04100000e+03 1.28000000e+03 1.44700000e+03 3.16600000e+03
     3.33300000e+03 1.76900000e+03 7.36000000e+02 1.96400000e+03
     1.61900000e+03 1.13000000e+04 1.45100000e+03 7.25000000e+03
     5.06300000e+03 2.13800000e+03 5.29600000e+03 2.58300000e+03
     2.36500000e+03 2.81600000e+03 2.50000000e+03 1.08300000e+03
     1.25000000e+03 3.02100000e+03 9.83000000e+02 1.80000000e+03
     1.77500000e+03 2.38300000e+03 1.71700000e+03 2.79100000e+03
     1.01000000e+03 1.69500000e+03 2.05400000e+03 2.59800000e+03
     1.77900000e+03 1.26000000e+03 5.00000000e+03 1.98300000e+03
     5.70100000e+03 1.30000000e+03 4.41700000e+03 4.33300000e+03
     1.84300000e+03 1.86800000e+03 3.89000000e+03 2.16700000e+03
     7.10100000e+03 2.10000000e+03 4.25000000e+03 2.20900000e+03
     3.44700000e+03 1.38700000e+03 1.81100000e+03 1.56000000e+03
     1.85700000e+03 2.22300000e+03 1.84200000e+03 3.27400000e+03
     2.42600000e+03 8.00000000e+02 9.85799988e+02 3.05300000e+03
     2.41600000e+03 3.33400000e+03 2.54100000e+03 2.93400000e+03
     1.75000000e+03 1.80300000e+03 1.86300000e+03 2.40500000e+03
     2.13400000e+03 1.89000000e+02 1.59000000e+03 2.98500000e+03
     4.98300000e+03 2.16000000e+03 2.45100000e+03 1.79300000e+03
     1.83300000e+03 4.49000000e+03 6.88000000e+02 4.60000000e+03
     1.58700000e+03 1.22900000e+03 2.33000000e+03 2.45800000e+03
     3.23000000e+03 2.16800000e+03 4.58300000e+03 6.25000000e+03
     5.05000000e+02 3.16700000e+03 3.66700000e+03 3.03300000e+03
     5.26600000e+03 7.87300000e+03 1.98700000e+03 9.23000000e+02
     4.99600000e+03 4.23200000e+03 1.60000000e+03 3.13600000e+03
     2.41700000e+03 2.11500000e+03 1.62500000e+03 1.40000000e+03
     4.84000000e+02 2.00000000e+04 2.40000000e+03 2.03300000e+03
     3.23700000e+03 2.77300000e+03 1.41700000e+03 1.71900000e+03
     4.30000000e+03 1.61200008e+01 2.34000000e+03 1.85100000e+03
     1.12500000e+03 5.06400000e+03 1.99300000e+03 8.33300000e+03
     1.21000000e+03 1.37600000e+03 1.71000000e+03 1.54200000e+03
     1.25500000e+03 1.45600000e+03 1.73300000e+03 2.46600000e+03
     4.08300000e+03 2.18800000e+03 1.66400000e+03 2.91700000e+03
     2.07900000e+03 1.50000000e+03 4.64800000e+03 1.01400000e+03
     1.87200000e+03 1.60300000e+03 3.15000000e+03 2.43600000e+03
     2.78500000e+03 1.13100000e+03 2.15700000e+03 9.13000000e+02
     1.70000000e+03 2.85700000e+03 4.41600000e+03 3.68300000e+03
     5.62400000e+03 5.30200000e+03 1.48300000e+03 6.66700000e+03
     3.01300000e+03 1.28700000e+03 2.00400000e+03 2.03500000e+03
     6.66600000e+03 3.66600000e+03 3.42800000e+03 1.63200000e+03
     1.91500000e+03 1.74200000e+03 1.42400000e+03 7.16600000e+03
     2.08700000e+03 1.30200000e+03 5.50000000e+03 2.04200000e+03
     3.90600000e+03 5.36000000e+02 2.84500000e+03 2.52400000e+03
     6.63000000e+02 1.95000000e+03 1.78300000e+03 2.01600000e+03
     2.37500000e+03 3.25000000e+03 4.26600000e+03 1.03200000e+03
     2.66900000e+03 2.30600000e+03 2.42000000e+02 2.06400000e+03
     4.61000000e+02 2.21000000e+03 2.73900000e+03 2.23200000e+03
     3.38370000e+04 1.52200000e+03 3.41600000e+03 3.30000000e+03
     1.00000000e+03 4.16670000e+04 2.79200000e+03 4.30100000e+03
     3.80000000e+03 1.41100000e+03 2.40000000e+02]
    [ nan 128.  66. 120. 141. 267.  95. 158. 168. 349.  70. 109. 200. 114.
      17. 125. 100.  76. 133. 115. 104. 315. 116. 112. 151. 191. 122. 110.
      35. 201.  74. 106. 320. 144. 184.  80.  47.  75. 134.  96.  88.  44.
     286.  97. 135. 180.  99. 165. 258. 126. 312. 136. 172.  81. 187. 113.
     176. 130. 111. 167. 265.  50. 210. 175. 131. 188.  25. 137. 160. 225.
     216.  94. 139. 152. 118. 185. 154.  85. 259. 194.  93. 370. 182. 650.
     102. 290.  84. 242. 129.  30. 244. 600. 255.  98. 275. 121.  63. 700.
      87. 101. 495.  67.  73. 260. 108.  58.  48. 164. 170.  83.  90. 166.
     124.  55.  59. 127. 214. 240.  72.  60. 138.  42. 280. 140. 155. 123.
     279. 192. 304. 330. 150. 207. 436.  78.  54.  89. 143. 105. 132. 480.
      56. 159. 300. 376. 117.  71. 490. 173.  46. 228. 308. 236. 570. 380.
     296. 156. 103.  45.  65.  53. 360.  62. 218. 178. 239. 405. 148. 190.
     149. 153. 162. 230.  86. 234. 246. 500. 186. 119. 107. 209. 208. 243.
      40. 250. 311. 400. 161. 196. 324. 157. 145. 181.  26. 211.   9. 205.
      36.  61. 146. 292. 142. 350. 496. 253.]
    [360. 120. 240.  nan 180.  60. 300. 480.  36.  84.  12.]
    [ 1.  0. nan]
    ['Urban' 'Rural' 'Semiurban']
    ['Y' 'N']
    


```python
data1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>614.000000</td>
      <td>614.000000</td>
      <td>592.000000</td>
      <td>600.00000</td>
      <td>564.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5403.459283</td>
      <td>1621.245798</td>
      <td>146.412162</td>
      <td>342.00000</td>
      <td>0.842199</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6109.041673</td>
      <td>2926.248369</td>
      <td>85.587325</td>
      <td>65.12041</td>
      <td>0.364878</td>
    </tr>
    <tr>
      <th>min</th>
      <td>150.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>12.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2877.500000</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3812.500000</td>
      <td>1188.500000</td>
      <td>128.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5795.000000</td>
      <td>2297.250000</td>
      <td>168.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81000.000000</td>
      <td>41667.000000</td>
      <td>700.000000</td>
      <td>480.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
data1.loc[:,'Dependents']=data1.loc[:,'Dependents'].replace("3+","3")
```


```python
ind= data1.loc[data1['Gender'].isna()==True].index
```


```python
print(ind)
```

    Index(['LP001050', 'LP001448', 'LP001585', 'LP001644', 'LP002024', 'LP002103',
           'LP002478', 'LP002501', 'LP002530', 'LP002625', 'LP002872', 'LP002925',
           'LP002933'],
          dtype='object', name='Loan_ID')
    


```python
data1=data1.drop(ind,axis=0)

```


```python
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 601 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             601 non-null    object 
     1   Married            598 non-null    object 
     2   Dependents         586 non-null    object 
     3   Education          601 non-null    object 
     4   Self_Employed      569 non-null    object 
     5   ApplicantIncome    601 non-null    int64  
     6   CoapplicantIncome  601 non-null    float64
     7   LoanAmount         579 non-null    float64
     8   Loan_Amount_Term   587 non-null    float64
     9   Credit_History     552 non-null    float64
     10  Property_Area      601 non-null    object 
     11  Loan_Status        601 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 61.0+ KB
    


```python
pd.to_datetime("bosto")
```


```python
ind= data1.loc[data1['Married'].isna()==True].index
```


```python
data1=data1.drop(ind,axis=0)
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 598 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             598 non-null    object 
     1   Married            598 non-null    object 
     2   Dependents         586 non-null    object 
     3   Education          598 non-null    object 
     4   Self_Employed      566 non-null    object 
     5   ApplicantIncome    598 non-null    int64  
     6   CoapplicantIncome  598 non-null    float64
     7   LoanAmount         577 non-null    float64
     8   Loan_Amount_Term   584 non-null    float64
     9   Credit_History     549 non-null    float64
     10  Property_Area      598 non-null    object 
     11  Loan_Status        598 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 60.7+ KB
    


```python
ind= data1.loc[data1['Dependents'].isna()==True].index
```


```python
data1=data1.drop(ind,axis=0)
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 586 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             586 non-null    object 
     1   Married            586 non-null    object 
     2   Dependents         586 non-null    object 
     3   Education          586 non-null    object 
     4   Self_Employed      554 non-null    object 
     5   ApplicantIncome    586 non-null    int64  
     6   CoapplicantIncome  586 non-null    float64
     7   LoanAmount         566 non-null    float64
     8   Loan_Amount_Term   573 non-null    float64
     9   Credit_History     537 non-null    float64
     10  Property_Area      586 non-null    object 
     11  Loan_Status        586 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 59.5+ KB
    


```python
ind= data1.loc[data1['Self_Employed'].isna()==True].index
```


```python
#data1=data1.drop(ind,axis=0)
for i in ind:
    data1.loc[i,'Self_Employed']="No info"
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 586 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             586 non-null    object 
     1   Married            586 non-null    object 
     2   Dependents         586 non-null    object 
     3   Education          586 non-null    object 
     4   Self_Employed      586 non-null    object 
     5   ApplicantIncome    586 non-null    int64  
     6   CoapplicantIncome  586 non-null    float64
     7   LoanAmount         566 non-null    float64
     8   Loan_Amount_Term   573 non-null    float64
     9   Credit_History     537 non-null    float64
     10  Property_Area      586 non-null    object 
     11  Loan_Status        586 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 79.5+ KB
    


```python
ind= data1.loc[data1['Loan_Amount_Term'].isna()==True].index
```


```python
for i in ind:
    data1.loc[i,'Loan_Amount_Term']=data1['Loan_Amount_Term'].mean()
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 586 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             586 non-null    object 
     1   Married            586 non-null    object 
     2   Dependents         586 non-null    object 
     3   Education          586 non-null    object 
     4   Self_Employed      586 non-null    object 
     5   ApplicantIncome    586 non-null    int64  
     6   CoapplicantIncome  586 non-null    float64
     7   LoanAmount         566 non-null    float64
     8   Loan_Amount_Term   586 non-null    float64
     9   Credit_History     537 non-null    float64
     10  Property_Area      586 non-null    object 
     11  Loan_Status        586 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 79.5+ KB
    


```python
ind= data1.loc[data1['LoanAmount'].isna()==True].index
```


```python
for i in ind:
    data1.loc[ind,'LoanAmount']=data1['LoanAmount'].mean()
#data1=data1.drop(ind,axis=0)
data1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 586 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             586 non-null    object 
     1   Married            586 non-null    object 
     2   Dependents         586 non-null    object 
     3   Education          586 non-null    object 
     4   Self_Employed      586 non-null    object 
     5   ApplicantIncome    586 non-null    int64  
     6   CoapplicantIncome  586 non-null    float64
     7   LoanAmount         586 non-null    float64
     8   Loan_Amount_Term   586 non-null    float64
     9   Credit_History     537 non-null    float64
     10  Property_Area      586 non-null    object 
     11  Loan_Status        586 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 79.5+ KB
    


```python
ind= data1.loc[data1['Credit_History'].isna()==True].index
```


```python
data1=data1.drop(ind,axis=0)
print(data1.loc[:,['Credit_History','Loan_Status']])
data1.info()
```

              Credit_History Loan_Status
    Loan_ID                             
    LP001002             1.0           Y
    LP001003             1.0           N
    LP001005             1.0           Y
    LP001006             1.0           Y
    LP001008             1.0           Y
    ...                  ...         ...
    LP002978             1.0           Y
    LP002979             1.0           Y
    LP002983             1.0           Y
    LP002984             1.0           Y
    LP002990             0.0           N
    
    [537 rows x 2 columns]
    <class 'pandas.core.frame.DataFrame'>
    Index: 537 entries, LP001002 to LP002990
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             537 non-null    object 
     1   Married            537 non-null    object 
     2   Dependents         537 non-null    object 
     3   Education          537 non-null    object 
     4   Self_Employed      537 non-null    object 
     5   ApplicantIncome    537 non-null    int64  
     6   CoapplicantIncome  537 non-null    float64
     7   LoanAmount         537 non-null    float64
     8   Loan_Amount_Term   537 non-null    float64
     9   Credit_History     537 non-null    float64
     10  Property_Area      537 non-null    object 
     11  Loan_Status        537 non-null    object 
    dtypes: float64(4), int64(1), object(7)
    memory usage: 54.5+ KB
    


```python
data_test=pd.read_csv('C:/Users/subha/OneDrive/Desktop/test_lAUu6dG.csv',header=0)
data_test=data_test.set_index('Loan_ID')
```


```python
data_test.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
    </tr>
    <tr>
      <th>Loan_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LP001015</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5720</td>
      <td>0</td>
      <td>110.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>LP001022</th>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3076</td>
      <td>1500</td>
      <td>126.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>LP001031</th>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5000</td>
      <td>1800</td>
      <td>208.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>LP001035</th>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2340</td>
      <td>2546</td>
      <td>100.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>LP001051</th>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3276</td>
      <td>0</td>
      <td>78.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 367 entries, LP001015 to LP002989
    Data columns (total 11 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             356 non-null    object 
     1   Married            367 non-null    object 
     2   Dependents         357 non-null    object 
     3   Education          367 non-null    object 
     4   Self_Employed      344 non-null    object 
     5   ApplicantIncome    367 non-null    int64  
     6   CoapplicantIncome  367 non-null    int64  
     7   LoanAmount         362 non-null    float64
     8   Loan_Amount_Term   361 non-null    float64
     9   Credit_History     338 non-null    float64
     10  Property_Area      367 non-null    object 
    dtypes: float64(3), int64(2), object(6)
    memory usage: 34.4+ KB
    


```python
sns.catplot(x="Loan_Status", hue="Gender", kind="count",palette="pastel", edgecolor=".6",data=data1)
```




    <seaborn.axisgrid.FacetGrid at 0x216a2496c08>




![png](output_30_1.png)



```python
sns.catplot(x="Loan_Status", hue="Married", kind="count",palette="pastel", edgecolor=".6",data=data1)
```




    <seaborn.axisgrid.FacetGrid at 0x216a34a56c8>




![png](output_31_1.png)



```python
categorical_data=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status','ApplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','CoapplicantIncome']
```


```python
lc=LabelEncoder()
```


```python
for i in categorical_data:
    data1.loc[:,i]=lc.fit_transform(data1.loc[:,i])
    
```


```python
print(data1['Loan_Status'])
```

    Loan_ID
    LP001002    1
    LP001003    0
    LP001005    1
    LP001006    1
    LP001008    1
               ..
    LP002978    1
    LP002979    1
    LP002983    1
    LP002984    1
    LP002990    0
    Name: Loan_Status, Length: 537, dtype: int32
    


```python
chi2(data1[categorical_data],data1['Loan_Status'])
```




    (array([1.99656925e-01, 1.61164512e+00, 2.06685363e-01, 3.04187317e+00,
            1.25939297e-04, 2.17149456e-01, 1.68000000e+02, 1.02200101e+01,
            2.16098459e+01, 2.11718591e-01, 2.41534031e+01, 2.69569962e+01]),
     array([6.54997909e-01, 2.04260703e-01, 6.49377881e-01, 8.11422357e-02,
            9.91046114e-01, 6.41220526e-01, 2.02302495e-38, 1.38925094e-03,
            3.34132183e-06, 6.45423712e-01, 8.89588303e-07, 2.08032780e-07]))




```python

```


```python
#Dropping all categorical data from chi2 test
#features=['ApplicantIncome','LoanAmount','Credit_History','CoapplicantIncome']
features=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','ApplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','CoapplicantIncome']
traindata_prediction =data1[features]
```


```python
traindata_prediction.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>Property_Area</th>
      <th>ApplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>CoapplicantIncome</th>
    </tr>
    <tr>
      <th>Loan_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LP001002</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>335</td>
      <td>96</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LP001003</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>274</td>
      <td>78</td>
      <td>9</td>
      <td>1</td>
      <td>55</td>
    </tr>
    <tr>
      <th>LP001005</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>126</td>
      <td>24</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LP001006</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>83</td>
      <td>70</td>
      <td>9</td>
      <td>1</td>
      <td>148</td>
    </tr>
    <tr>
      <th>LP001008</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>340</td>
      <td>91</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc=StandardScaler()
```


```python
traindata_prediction=sc.fit_transform(traindata_prediction)
```


```python
target=data1['Loan_Status']
```


```python
X_train,X_test,Y_train,Y_test=train_test_split(traindata_prediction,target,test_size=0.20,random_state=42)
```


```python
model=GradientBoostingClassifier()
model.fit(X_train,Y_train)

```




    GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=3,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_iter_no_change=None, presort='deprecated',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)




```python
y_pred=model.predict(X_test)
```


```python
print(accuracy_score(Y_test,y_pred))
```

    0.8055555555555556
    


```python
data_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 367 entries, LP001015 to LP002989
    Data columns (total 11 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             356 non-null    object 
     1   Married            367 non-null    object 
     2   Dependents         357 non-null    object 
     3   Education          367 non-null    object 
     4   Self_Employed      344 non-null    object 
     5   ApplicantIncome    367 non-null    int64  
     6   CoapplicantIncome  367 non-null    int64  
     7   LoanAmount         362 non-null    float64
     8   Loan_Amount_Term   361 non-null    float64
     9   Credit_History     338 non-null    float64
     10  Property_Area      367 non-null    object 
    dtypes: float64(3), int64(2), object(6)
    memory usage: 34.4+ KB
    


```python
testdata=['ApplicantIncome','LoanAmount','Credit_History','CoapplicantIncome','Loan_Amount_Term']
data_test[testdata].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ApplicantIncome</th>
      <th>LoanAmount</th>
      <th>Credit_History</th>
      <th>CoapplicantIncome</th>
      <th>Loan_Amount_Term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>367.000000</td>
      <td>362.000000</td>
      <td>338.000000</td>
      <td>367.000000</td>
      <td>361.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4805.599455</td>
      <td>136.132597</td>
      <td>0.825444</td>
      <td>1569.577657</td>
      <td>342.537396</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4910.685399</td>
      <td>61.366652</td>
      <td>0.380150</td>
      <td>2334.232099</td>
      <td>65.156643</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2864.000000</td>
      <td>100.250000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>360.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3786.000000</td>
      <td>125.000000</td>
      <td>1.000000</td>
      <td>1025.000000</td>
      <td>360.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5060.000000</td>
      <td>158.000000</td>
      <td>1.000000</td>
      <td>2430.500000</td>
      <td>360.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>72529.000000</td>
      <td>550.000000</td>
      <td>1.000000</td>
      <td>24000.000000</td>
      <td>480.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for i in ['Gender','Married','Dependents','Self_Employed','Education','Property_Area']:
    ind=data_test.loc[data_test[i].isna()==True].index
    for j in ind:
     data_test.loc[ind,i]="No info"
#data1=data1.drop(ind,axis=0)
data_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 367 entries, LP001015 to LP002989
    Data columns (total 11 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             367 non-null    object 
     1   Married            367 non-null    object 
     2   Dependents         367 non-null    object 
     3   Education          367 non-null    object 
     4   Self_Employed      367 non-null    object 
     5   ApplicantIncome    367 non-null    int64  
     6   CoapplicantIncome  367 non-null    int64  
     7   LoanAmount         362 non-null    float64
     8   Loan_Amount_Term   361 non-null    float64
     9   Credit_History     338 non-null    float64
     10  Property_Area      367 non-null    object 
    dtypes: float64(3), int64(2), object(6)
    memory usage: 44.4+ KB
    


```python
for i in ['Gender','Married','Dependents','Self_Employed','Education','Property_Area']:
  data_test.loc[:,i]=lc.fit_transform(data_test.loc[:,i])
```


```python
for i in testdata:
    ind= data_test.loc[data_test[i].isna()==True].index
    for j in ind:
        data_test.loc[j,i]=data_test[i].mean()
```


```python
data_test[testdata]=sc.fit_transform(data_test[testdata])
```


```python
data_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 367 entries, LP001015 to LP002989
    Data columns (total 11 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Gender             367 non-null    int32  
     1   Married            367 non-null    int32  
     2   Dependents         367 non-null    int32  
     3   Education          367 non-null    int32  
     4   Self_Employed      367 non-null    int32  
     5   ApplicantIncome    367 non-null    float64
     6   CoapplicantIncome  367 non-null    float64
     7   LoanAmount         367 non-null    float64
     8   Loan_Amount_Term   367 non-null    float64
     9   Credit_History     367 non-null    float64
     10  Property_Area      367 non-null    int32  
    dtypes: float64(5), int32(6)
    memory usage: 35.8+ KB
    


```python
predicted_value=model.predict(data_test)
predicted_value=list(predicted_value)
output=[]
```


```python
for i in range(len(predicted_value)):
    if(predicted_value[i]==0):
        output.append('N')
    elif(predicted_value[i]==1):
        output.append('Y')
print(len(output))
data_test.reset_index(inplace=True) 
data_test['Status']=output
testval=pd.DataFrame({'Loan_ID':data_test['Loan_ID'],'Loan_Status':data_test['Status']})
testval.to_csv('Submission.csv',index=False)
```

    367
    


```python

```


```python

```


```python

```

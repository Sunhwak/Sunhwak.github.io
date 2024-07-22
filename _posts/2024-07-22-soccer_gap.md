```python
!jupyter nbconvert --to markdown "/content/drive/MyDrive/DAT/5차 과제"
```

    [NbConvertApp] Converting notebook /content/drive/MyDrive/DAT/5차 과제 to markdown
    Traceback (most recent call last):
      File "/usr/local/bin/jupyter-nbconvert", line 8, in <module>
        sys.exit(main())
      File "/usr/local/lib/python3.10/dist-packages/jupyter_core/application.py", line 283, in launch_instance
        super().launch_instance(argv=argv, **kwargs)
      File "/usr/local/lib/python3.10/dist-packages/traitlets/config/application.py", line 992, in launch_instance
        app.start()
      File "/usr/local/lib/python3.10/dist-packages/nbconvert/nbconvertapp.py", line 423, in start
        self.convert_notebooks()
      File "/usr/local/lib/python3.10/dist-packages/nbconvert/nbconvertapp.py", line 597, in convert_notebooks
        self.convert_single_notebook(notebook_filename)
      File "/usr/local/lib/python3.10/dist-packages/nbconvert/nbconvertapp.py", line 560, in convert_single_notebook
        output, resources = self.export_single_notebook(
      File "/usr/local/lib/python3.10/dist-packages/nbconvert/nbconvertapp.py", line 488, in export_single_notebook
        output, resources = self.exporter.from_filename(
      File "/usr/local/lib/python3.10/dist-packages/nbconvert/exporters/exporter.py", line 188, in from_filename
        with open(filename, encoding="utf-8") as f:
    IsADirectoryError: [Errno 21] Is a directory: '/content/drive/MyDrive/DAT/5차 과제'



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## 변수 처리


```python
df = pd.read_csv('league_5_GAP.csv')

df.head()
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
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTR</th>
      <th>AtGAP</th>
      <th>MidGAP</th>
      <th>DefGAP</th>
      <th>SquadGAP</th>
      <th>AgeGAP</th>
      <th>MVGAP</th>
      <th>...</th>
      <th>SGAP</th>
      <th>STGAP</th>
      <th>FGAP</th>
      <th>CGAP</th>
      <th>YGAP</th>
      <th>RGAP</th>
      <th>xGGAP</th>
      <th>xAGAP</th>
      <th>xPTSGAP</th>
      <th>PPDAGAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-09-26</td>
      <td>Elche</td>
      <td>Celta Vigo</td>
      <td>A</td>
      <td>-4.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-21.800000</td>
      <td>...</td>
      <td>-0.6</td>
      <td>-2.0</td>
      <td>-1.8</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-0.708</td>
      <td>-0.518</td>
      <td>-0.852</td>
      <td>5.982</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-27</td>
      <td>Atletico Madrid</td>
      <td>Sevilla</td>
      <td>H</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>189.250000</td>
      <td>...</td>
      <td>0.2</td>
      <td>-0.4</td>
      <td>-5.6</td>
      <td>-0.8</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>-0.346</td>
      <td>0.294</td>
      <td>-0.194</td>
      <td>3.558</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-09-27</td>
      <td>Villarreal</td>
      <td>Real Madrid</td>
      <td>A</td>
      <td>-9.0</td>
      <td>-10.0</td>
      <td>-8.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-656.100001</td>
      <td>...</td>
      <td>-5.4</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>-0.146</td>
      <td>-0.030</td>
      <td>-0.050</td>
      <td>4.382</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-09-27</td>
      <td>Athletic Club</td>
      <td>Eibar</td>
      <td>D</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>82.700000</td>
      <td>...</td>
      <td>-0.6</td>
      <td>0.8</td>
      <td>-0.4</td>
      <td>-1.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.316</td>
      <td>0.236</td>
      <td>0.006</td>
      <td>-1.228</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-09-27</td>
      <td>Barcelona</td>
      <td>Granada</td>
      <td>H</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>-8.0</td>
      <td>0.0</td>
      <td>536.250000</td>
      <td>...</td>
      <td>7.8</td>
      <td>3.6</td>
      <td>-4.0</td>
      <td>0.8</td>
      <td>-1.4</td>
      <td>0.2</td>
      <td>1.712</td>
      <td>1.114</td>
      <td>1.322</td>
      <td>-2.662</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df = df.replace({'FTR' : 'H'}, 0)
df = df.replace({'FTR' : 'A'}, 1)
df = df.replace({'FTR' : 'D'}, 2)
```


```python
df = df.dropna()
```


```python

```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12548 entries, 0 to 12547
    Data columns (total 22 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Date      12548 non-null  object 
     1   HomeTeam  12548 non-null  object 
     2   AwayTeam  12548 non-null  object 
     3   FTR       12548 non-null  int64  
     4   AtGAP     12548 non-null  float64
     5   MidGAP    12548 non-null  float64
     6   DefGAP    12548 non-null  float64
     7   SquadGAP  12548 non-null  float64
     8   AgeGAP    12548 non-null  float64
     9   MVGAP     12548 non-null  float64
     10  FTG       12548 non-null  float64
     11  HTG       12548 non-null  float64
     12  SGAP      12548 non-null  float64
     13  STGAP     12548 non-null  float64
     14  FGAP      12548 non-null  float64
     15  CGAP      12548 non-null  float64
     16  YGAP      12548 non-null  float64
     17  RGAP      12548 non-null  float64
     18  xGGAP     12548 non-null  float64
     19  xAGAP     12548 non-null  float64
     20  xPTSGAP   12548 non-null  float64
     21  PPDAGAP   12548 non-null  float64
    dtypes: float64(18), int64(1), object(3)
    memory usage: 2.1+ MB



```python
df[df.isna( ).any(axis=1)]
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
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTR</th>
      <th>AtGAP</th>
      <th>MidGAP</th>
      <th>DefGAP</th>
      <th>SquadGAP</th>
      <th>AgeGAP</th>
      <th>MVGAP</th>
      <th>...</th>
      <th>SGAP</th>
      <th>STGAP</th>
      <th>FGAP</th>
      <th>CGAP</th>
      <th>YGAP</th>
      <th>RGAP</th>
      <th>xGGAP</th>
      <th>xAGAP</th>
      <th>xPTSGAP</th>
      <th>PPDAGAP</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 22 columns</p>
</div>




```python
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
```


```python
df
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
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTR</th>
      <th>AtGAP</th>
      <th>MidGAP</th>
      <th>DefGAP</th>
      <th>SquadGAP</th>
      <th>AgeGAP</th>
      <th>MVGAP</th>
      <th>...</th>
      <th>STGAP</th>
      <th>FGAP</th>
      <th>CGAP</th>
      <th>YGAP</th>
      <th>RGAP</th>
      <th>xGGAP</th>
      <th>xAGAP</th>
      <th>xPTSGAP</th>
      <th>PPDAGAP</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-09-26</td>
      <td>Elche</td>
      <td>Celta Vigo</td>
      <td>1</td>
      <td>-4.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-21.800000</td>
      <td>...</td>
      <td>-2.0</td>
      <td>-1.8</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-0.708</td>
      <td>-0.518</td>
      <td>-0.852</td>
      <td>5.982</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-27</td>
      <td>Atletico Madrid</td>
      <td>Sevilla</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>189.250000</td>
      <td>...</td>
      <td>-0.4</td>
      <td>-5.6</td>
      <td>-0.8</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>-0.346</td>
      <td>0.294</td>
      <td>-0.194</td>
      <td>3.558</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-09-27</td>
      <td>Villarreal</td>
      <td>Real Madrid</td>
      <td>1</td>
      <td>-9.0</td>
      <td>-10.0</td>
      <td>-8.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-656.100001</td>
      <td>...</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>-0.146</td>
      <td>-0.030</td>
      <td>-0.050</td>
      <td>4.382</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-09-27</td>
      <td>Athletic Club</td>
      <td>Eibar</td>
      <td>2</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>82.700000</td>
      <td>...</td>
      <td>0.8</td>
      <td>-0.4</td>
      <td>-1.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.316</td>
      <td>0.236</td>
      <td>0.006</td>
      <td>-1.228</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-09-27</td>
      <td>Barcelona</td>
      <td>Granada</td>
      <td>0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>-8.0</td>
      <td>0.0</td>
      <td>536.250000</td>
      <td>...</td>
      <td>3.6</td>
      <td>-4.0</td>
      <td>0.8</td>
      <td>-1.4</td>
      <td>0.2</td>
      <td>1.712</td>
      <td>1.114</td>
      <td>1.322</td>
      <td>-2.662</td>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12543</th>
      <td>2021-09-25</td>
      <td>Union Berlin</td>
      <td>Arminia Bielefeld</td>
      <td>0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>50.050001</td>
      <td>...</td>
      <td>1.8</td>
      <td>-2.8</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.658</td>
      <td>0.484</td>
      <td>1.120</td>
      <td>2.352</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12544</th>
      <td>2021-09-25</td>
      <td>Bayer Leverkusen</td>
      <td>Mainz 05</td>
      <td>0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>324.020000</td>
      <td>...</td>
      <td>0.6</td>
      <td>-7.8</td>
      <td>-0.2</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>-0.030</td>
      <td>0.310</td>
      <td>-0.614</td>
      <td>1.318</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12545</th>
      <td>2021-09-25</td>
      <td>Hoffenheim</td>
      <td>Wolfsburg</td>
      <td>0</td>
      <td>-3.0</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-57.870000</td>
      <td>...</td>
      <td>-1.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>-0.4</td>
      <td>0.0</td>
      <td>0.226</td>
      <td>0.696</td>
      <td>-0.548</td>
      <td>3.474</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12546</th>
      <td>2021-09-25</td>
      <td>Eintracht Frankfurt</td>
      <td>FC Cologne</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>129.470000</td>
      <td>...</td>
      <td>-1.2</td>
      <td>1.4</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>-0.2</td>
      <td>-0.490</td>
      <td>-0.380</td>
      <td>0.008</td>
      <td>0.534</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12547</th>
      <td>2021-09-24</td>
      <td>Greuther Fuerth</td>
      <td>Bayern Munich</td>
      <td>1</td>
      <td>-20.0</td>
      <td>-13.0</td>
      <td>-11.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-753.380000</td>
      <td>...</td>
      <td>-5.4</td>
      <td>4.8</td>
      <td>-1.8</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>-3.062</td>
      <td>-2.640</td>
      <td>-1.794</td>
      <td>2.862</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>12548 rows × 23 columns</p>
</div>




```python

```


```python
df = df.drop(['Date'], axis = 1)
```


```python
df
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
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTR</th>
      <th>AtGAP</th>
      <th>MidGAP</th>
      <th>DefGAP</th>
      <th>SquadGAP</th>
      <th>AgeGAP</th>
      <th>MVGAP</th>
      <th>FTG</th>
      <th>...</th>
      <th>STGAP</th>
      <th>FGAP</th>
      <th>CGAP</th>
      <th>YGAP</th>
      <th>RGAP</th>
      <th>xGGAP</th>
      <th>xAGAP</th>
      <th>xPTSGAP</th>
      <th>PPDAGAP</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Elche</td>
      <td>Celta Vigo</td>
      <td>1</td>
      <td>-4.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-21.800000</td>
      <td>-1.0</td>
      <td>...</td>
      <td>-2.0</td>
      <td>-1.8</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-0.708</td>
      <td>-0.518</td>
      <td>-0.852</td>
      <td>5.982</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Atletico Madrid</td>
      <td>Sevilla</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>189.250000</td>
      <td>-0.4</td>
      <td>...</td>
      <td>-0.4</td>
      <td>-5.6</td>
      <td>-0.8</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>-0.346</td>
      <td>0.294</td>
      <td>-0.194</td>
      <td>3.558</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Villarreal</td>
      <td>Real Madrid</td>
      <td>1</td>
      <td>-9.0</td>
      <td>-10.0</td>
      <td>-8.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-656.100001</td>
      <td>-2.2</td>
      <td>...</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>-0.146</td>
      <td>-0.030</td>
      <td>-0.050</td>
      <td>4.382</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Athletic Club</td>
      <td>Eibar</td>
      <td>2</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>82.700000</td>
      <td>-0.2</td>
      <td>...</td>
      <td>0.8</td>
      <td>-0.4</td>
      <td>-1.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.316</td>
      <td>0.236</td>
      <td>0.006</td>
      <td>-1.228</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barcelona</td>
      <td>Granada</td>
      <td>0</td>
      <td>16.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>-8.0</td>
      <td>0.0</td>
      <td>536.250000</td>
      <td>1.4</td>
      <td>...</td>
      <td>3.6</td>
      <td>-4.0</td>
      <td>0.8</td>
      <td>-1.4</td>
      <td>0.2</td>
      <td>1.712</td>
      <td>1.114</td>
      <td>1.322</td>
      <td>-2.662</td>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12543</th>
      <td>Union Berlin</td>
      <td>Arminia Bielefeld</td>
      <td>0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>50.050001</td>
      <td>0.8</td>
      <td>...</td>
      <td>1.8</td>
      <td>-2.8</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.658</td>
      <td>0.484</td>
      <td>1.120</td>
      <td>2.352</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12544</th>
      <td>Bayer Leverkusen</td>
      <td>Mainz 05</td>
      <td>0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>324.020000</td>
      <td>1.8</td>
      <td>...</td>
      <td>0.6</td>
      <td>-7.8</td>
      <td>-0.2</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>-0.030</td>
      <td>0.310</td>
      <td>-0.614</td>
      <td>1.318</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12545</th>
      <td>Hoffenheim</td>
      <td>Wolfsburg</td>
      <td>0</td>
      <td>-3.0</td>
      <td>-2.0</td>
      <td>-2.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-57.870000</td>
      <td>0.2</td>
      <td>...</td>
      <td>-1.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>-0.4</td>
      <td>0.0</td>
      <td>0.226</td>
      <td>0.696</td>
      <td>-0.548</td>
      <td>3.474</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12546</th>
      <td>Eintracht Frankfurt</td>
      <td>FC Cologne</td>
      <td>2</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>129.470000</td>
      <td>-0.8</td>
      <td>...</td>
      <td>-1.2</td>
      <td>1.4</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>-0.2</td>
      <td>-0.490</td>
      <td>-0.380</td>
      <td>0.008</td>
      <td>0.534</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12547</th>
      <td>Greuther Fuerth</td>
      <td>Bayern Munich</td>
      <td>1</td>
      <td>-20.0</td>
      <td>-13.0</td>
      <td>-11.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-753.380000</td>
      <td>-3.4</td>
      <td>...</td>
      <td>-5.4</td>
      <td>4.8</td>
      <td>-1.8</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>-3.062</td>
      <td>-2.640</td>
      <td>-1.794</td>
      <td>2.862</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>12548 rows × 22 columns</p>
</div>




```python
# df.loc[df['HR'] >= 1, 'HR'] = 1 # 레드카드의 유무로 변경 (있으면 1 / 없으면 0)
# df.loc[df['HR'] < 1, 'HR'] = 0

# df
```


```python
# df.loc[df['AR'] >= 1, 'AR'] = 1
# df.loc[df['AR'] < 1, 'AR'] = 0

# df
```


```python
# print(df['HR'].unique())
# print(df['AR'].unique())
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12548 entries, 0 to 12547
    Data columns (total 22 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   HomeTeam  12548 non-null  object 
     1   AwayTeam  12548 non-null  object 
     2   FTR       12548 non-null  int64  
     3   AtGAP     12548 non-null  float64
     4   MidGAP    12548 non-null  float64
     5   DefGAP    12548 non-null  float64
     6   SquadGAP  12548 non-null  float64
     7   AgeGAP    12548 non-null  float64
     8   MVGAP     12548 non-null  float64
     9   FTG       12548 non-null  float64
     10  HTG       12548 non-null  float64
     11  SGAP      12548 non-null  float64
     12  STGAP     12548 non-null  float64
     13  FGAP      12548 non-null  float64
     14  CGAP      12548 non-null  float64
     15  YGAP      12548 non-null  float64
     16  RGAP      12548 non-null  float64
     17  xGGAP     12548 non-null  float64
     18  xAGAP     12548 non-null  float64
     19  xPTSGAP   12548 non-null  float64
     20  PPDAGAP   12548 non-null  float64
     21  month     12548 non-null  int64  
    dtypes: float64(18), int64(2), object(2)
    memory usage: 2.1+ MB



```python
df[df.isna( ).any(axis=1)]
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
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTR</th>
      <th>AtGAP</th>
      <th>MidGAP</th>
      <th>DefGAP</th>
      <th>SquadGAP</th>
      <th>AgeGAP</th>
      <th>MVGAP</th>
      <th>FTG</th>
      <th>...</th>
      <th>STGAP</th>
      <th>FGAP</th>
      <th>CGAP</th>
      <th>YGAP</th>
      <th>RGAP</th>
      <th>xGGAP</th>
      <th>xAGAP</th>
      <th>xPTSGAP</th>
      <th>PPDAGAP</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 22 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12548 entries, 0 to 12547
    Data columns (total 22 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   HomeTeam  12548 non-null  object 
     1   AwayTeam  12548 non-null  object 
     2   FTR       12548 non-null  int64  
     3   AtGAP     12548 non-null  float64
     4   MidGAP    12548 non-null  float64
     5   DefGAP    12548 non-null  float64
     6   SquadGAP  12548 non-null  float64
     7   AgeGAP    12548 non-null  float64
     8   MVGAP     12548 non-null  float64
     9   FTG       12548 non-null  float64
     10  HTG       12548 non-null  float64
     11  SGAP      12548 non-null  float64
     12  STGAP     12548 non-null  float64
     13  FGAP      12548 non-null  float64
     14  CGAP      12548 non-null  float64
     15  YGAP      12548 non-null  float64
     16  RGAP      12548 non-null  float64
     17  xGGAP     12548 non-null  float64
     18  xAGAP     12548 non-null  float64
     19  xPTSGAP   12548 non-null  float64
     20  PPDAGAP   12548 non-null  float64
     21  month     12548 non-null  int64  
    dtypes: float64(18), int64(2), object(2)
    memory usage: 2.1+ MB


## 모델링




```python
df_target = df['FTR'].to_numpy()
df_home = df['HomeTeam']
df_away = df['AwayTeam']
```


```python
df_input = df.drop(['FTR', 'HomeTeam', 'AwayTeam'], axis=1)
```


```python
df_target
```




    array([1, 0, 1, ..., 0, 2, 1])




```python
df_input.to_numpy()
```




    array([[-4.000e+00, -3.000e+00,  0.000e+00, ..., -8.520e-01,  5.982e+00,
             9.000e+00],
           [ 0.000e+00,  3.000e+00,  2.000e+00, ..., -1.940e-01,  3.558e+00,
             9.000e+00],
           [-9.000e+00, -1.000e+01, -8.000e+00, ..., -5.000e-02,  4.382e+00,
             9.000e+00],
           ...,
           [-3.000e+00, -2.000e+00, -2.000e+00, ..., -5.480e-01,  3.474e+00,
             9.000e+00],
           [ 2.000e+00,  3.000e+00,  4.000e+00, ...,  8.000e-03,  5.340e-01,
             9.000e+00],
           [-2.000e+01, -1.300e+01, -1.100e+01, ..., -1.794e+00,  2.862e+00,
             9.000e+00]])




```python
df_input[:5]
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
      <th>AtGAP</th>
      <th>MidGAP</th>
      <th>DefGAP</th>
      <th>SquadGAP</th>
      <th>AgeGAP</th>
      <th>MVGAP</th>
      <th>FTG</th>
      <th>HTG</th>
      <th>SGAP</th>
      <th>STGAP</th>
      <th>FGAP</th>
      <th>CGAP</th>
      <th>YGAP</th>
      <th>RGAP</th>
      <th>xGGAP</th>
      <th>xAGAP</th>
      <th>xPTSGAP</th>
      <th>PPDAGAP</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-4.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-21.800000</td>
      <td>-1.0</td>
      <td>-0.4</td>
      <td>-0.6</td>
      <td>-2.0</td>
      <td>-1.8</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-0.708</td>
      <td>-0.518</td>
      <td>-0.852</td>
      <td>5.982</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>189.250000</td>
      <td>-0.4</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>-0.4</td>
      <td>-5.6</td>
      <td>-0.8</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>-0.346</td>
      <td>0.294</td>
      <td>-0.194</td>
      <td>3.558</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-9.0</td>
      <td>-10.0</td>
      <td>-8.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>-656.100001</td>
      <td>-2.2</td>
      <td>-1.8</td>
      <td>-5.4</td>
      <td>-4.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>-0.146</td>
      <td>-0.030</td>
      <td>-0.050</td>
      <td>4.382</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>-2.0</td>
      <td>82.700000</td>
      <td>-0.2</td>
      <td>-0.6</td>
      <td>-0.6</td>
      <td>0.8</td>
      <td>-0.4</td>
      <td>-1.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.316</td>
      <td>0.236</td>
      <td>0.006</td>
      <td>-1.228</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.0</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>-8.0</td>
      <td>0.0</td>
      <td>536.250000</td>
      <td>1.4</td>
      <td>0.4</td>
      <td>7.8</td>
      <td>3.6</td>
      <td>-4.0</td>
      <td>0.8</td>
      <td>-1.4</td>
      <td>0.2</td>
      <td>1.712</td>
      <td>1.114</td>
      <td>1.322</td>
      <td>-2.662</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split # 훈련 세트, 테스트 세트 나누기

train_input, test_input, train_target, test_target = train_test_split(df_input, df_target, test_size = 0.25, random_state=42)
```


```python
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```


```python
train_scaled.shape
```




    (9411, 19)




```python
train_target.shape
```




    (9411,)




```python
test_scaled.shape
```




    (3137, 19)




```python
test_target.shape
```




    (3137,)




```python

```


```python
#모델 훈련 후 정확도 측정
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3) # 참고하는 이웃의 수를 3으로 지정
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

    0.6858994793326958
    0.45616831367548616



```python
print(kn.classes_)
```

    [0 1 2]



```python
#모델 예측값 알아보기
print(kn.predict(test_scaled[:5]))
```

    [2 0 0 0 1]



```python
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

```

    [[0.3333 0.     0.6667]
     [0.6667 0.3333 0.    ]
     [0.3333 0.3333 0.3333]
     [1.     0.     0.    ]
     [0.     1.     0.    ]]



```python
from sklearn.neighbors import KNeighborsClassifier

kn49 = KNeighborsClassifier(n_neighbors=49) # 참고하는 이웃의 수를 49로 지정
kn49.fit(train_scaled, train_target)
print(kn49.score(train_scaled, train_target))
print(kn49.score(test_scaled, test_target))
```

    0.5424503240888322
    0.49824673254701946



```python
from sklearn.neighbors import KNeighborsClassifier

kn100 = KNeighborsClassifier(n_neighbors=100) # 참고하는 이웃의 수를 100으로 지정
kn100.fit(train_scaled, train_target)
print(kn100.score(train_scaled, train_target))
print(kn100.score(test_scaled, test_target))
```

    0.5367123578790777
    0.5106789926681543



```python
from sklearn.neighbors import KNeighborsClassifier

kn150 = KNeighborsClassifier(n_neighbors=150) # 참고하는 이웃의 수를 150으로 지정
kn150.fit(train_scaled, train_target)
print(kn150.score(train_scaled, train_target))
print(kn150.score(test_scaled, test_target))
```

    0.5327807884390606
    0.5116353203697801



```python
from sklearn.neighbors import KNeighborsClassifier

kn149 = KNeighborsClassifier(n_neighbors=149) # 참고하는 이웃의 수를 150으로 지정
kn149.fit(train_scaled, train_target)
print(kn149.score(train_scaled, train_target))
print(kn149.score(test_scaled, test_target))
```

    0.5319307193709489
    0.5106789926681543



```python
from sklearn.neighbors import KNeighborsClassifier

kn147 = KNeighborsClassifier(n_neighbors=147) # 참고하는 이웃의 수를 150으로 지정
kn147.fit(train_scaled, train_target)
print(kn147.score(train_scaled, train_target))
print(kn147.score(test_scaled, test_target))
```

    0.5310806503028371
    0.5109977685686962





```python
from sklearn.neighbors import KNeighborsClassifier

kn160 = KNeighborsClassifier(n_neighbors=160) # 참고하는 이웃의 수를 150으로 지정
kn160.fit(train_scaled, train_target)
print(kn160.score(train_scaled, train_target))
print(kn160.score(test_scaled, test_target))
```

    0.531505684836893
    0.5106789926681543



```python
from sklearn.neighbors import KNeighborsClassifier

kn151 = KNeighborsClassifier(n_neighbors=151) # 참고하는 이웃의 수를 150으로 지정
kn151.fit(train_scaled, train_target)
print(kn151.score(train_scaled, train_target))
print(kn151.score(test_scaled, test_target))
```

    0.531611943470407
    0.5109977685686962



```python
from sklearn.neighbors import KNeighborsClassifier

kn195 = KNeighborsClassifier(n_neighbors=195)
kn195.fit(train_scaled, train_target)
print(kn195.score(train_scaled, train_target))
print(kn195.score(test_scaled, test_target))
```

    0.5322494952714908
    0.5094038890659867



```python
print(kn195.predict_proba(test_scaled[:5]))
```

    [[0.46153846 0.26666667 0.27179487]
     [0.55897436 0.18461538 0.25641026]
     [0.36410256 0.32307692 0.31282051]
     [0.82564103 0.0974359  0.07692308]
     [0.41538462 0.30769231 0.27692308]]



```python
#모델 예측값 알아보기
print(kn150.predict(test_scaled[:5]))

# 실제는 AHADH
#      10120
```

    [0 0 0 0 0]



```python
y_pred = kn150.predict(test_scaled)
```


```python
y_pred
```




    array([0, 0, 0, ..., 0, 0, 0])




```python

```


```python
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(test_target, y_pred)
print('Accuracy :', accuracy)
```

    Accuracy : 0.5116353203697801



```python
f1 = f1_score(test_target, y_pred, average='weighted')
print("F1 Score:", f1)
```

    F1 Score: 0.4347208870476543



```python

```

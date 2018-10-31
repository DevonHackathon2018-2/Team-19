import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from sklearn.externals import joblib


validation_df_location="/mnt/data/test/Warzecha__P__A_2_4_CMP_10-5-2018.csv"
train_ds = "/mnt/data/compressors_eagleford_travis_20181025142853.txt"
model_name = 'compressors_model.joblib'

df = pd.read_csv(train_ds, sep="\t", dtype={'Pct Successful Msgs Today': 'float64',
                                                                                  'RPM': 'float64',
                                                                                  'Successful Msgs Today': 'float64'})

df_validation = pd.read_csv(validation_df_location, sep=",", dtype={'Pct Successful Msgs Today': 'float64',
                                                                    'RPM': 'object',
                                                                    'Successful Msgs Today': 'float64',
                                                                    'Pct Successful Msgs Today': 'object'})


print("STARTED_PROCESSING  , validation data at %s " % validation_df_location)


df_validation = df_validation.fillna(0)
df_validation = df_validation.replace({'No Data': 0.0, 'Bad': '0.0'})
df_validation['RPM'] = df_validation['RPM'].apply(lambda col:pd.to_numeric(col, errors='coerce'))

columns = ['Id', 'Asset Name', 'Local Timestamp', 'UTC Milliseconds',
           'Compressor Oil Pressure', 'Compressor Oil Temp', 'Compressor Stages',
           'Cylinder 1 Discharge Temp', 'Cylinder 2 Discharge Temp',
           'Cylinder 3 Discharge Temp', 'Cylinder 4 Discharge Temp',
           'Downtime Hrs Yest', 'Engine Oil Pressure', 'Engine Oil Temp',
           'Facility Desc', 'Facility ID', 'Fuel Pressure', 'Gas Flow Rate',
           'Gas Flow Rate_RAW', 'Horsepower', 'Last Successful Comm Time',
           'Max Discharge Pressure', 'Max Gas Flowrate', 'Max RPMs',
           'Max Suction Pressure', 'Pct Successful Msgs Today', 'RPM',
           'Run Status', 'Runtime Hrs', 'SD Status Code',
           'Stage 1 Discharge Pressure', 'Stage 2 Discharge Pressure',
           'Stage 3 Discharge Pressure', 'Successful Msgs Today',
           'Suction Pressure', 'Suction Temp', 'TOW Comp Name', 'Unit Size']

tv_columns = ['Id', 'Asset Name', 'Local Timestamp',
              'Compressor Oil Pressure', 'Compressor Oil Temp', 'Compressor Stages',
              'Cylinder 1 Discharge Temp', 'Cylinder 2 Discharge Temp',
              'Cylinder 3 Discharge Temp', 'Cylinder 4 Discharge Temp',
              'Downtime Hrs Yest', 'Engine Oil Pressure', 'Engine Oil Temp',
              'Facility Desc', 'Facility ID', 'Fuel Pressure', 'Gas Flow Rate',
              'Gas Flow Rate_RAW', 'Horsepower', 'Last Successful Comm Time',
              'Max Discharge Pressure', 'Max Gas Flowrate', 'Max RPMs',
              'Max Suction Pressure', 'Pct Successful Msgs Today', 'RPM',
              'Run Status', 'Runtime Hrs', 'SD Status Code',
              'Stage 1 Discharge Pressure', 'Stage 2 Discharge Pressure',
              'Stage 3 Discharge Pressure', 'Successful Msgs Today',
              'Suction Pressure', 'Suction Temp', 'TOW Comp Name', 'Unit Size']

PRESSURE_INDICATORS = ['Compressor Oil Pressure',
                       'Engine Oil Pressure',
                       'Fuel Pressure',
                       'Max Discharge Pressure',
                       'Max Suction Pressure',
                       'Stage 1 Discharge Pressure', 'Stage 2 Discharge Pressure',
                       'Stage 3 Discharge Pressure',
                       'Suction Pressure']

TEMPERATURE_INDICATORS = ['Compressor Oil Temp',
                          'Cylinder 1 Discharge Temp', 'Cylinder 2 Discharge Temp',
                          'Cylinder 3 Discharge Temp', 'Cylinder 4 Discharge Temp',
                          'Engine Oil Temp', 'Suction Temp']

OTHER_INDICATORS = ['Compressor Stages',
                    'Downtime Hrs Yest', 'Gas Flow Rate',
                    'Gas Flow Rate_RAW', 'Horsepower', 'Last Successful Comm Time',
                    'Max Gas Flowrate', 'Max RPMs',
                    'Pct Successful Msgs Today', 'RPM',
                    'Runtime Hrs']

failure_status = set(
    ["CAT FAIL TO STOP", "CAT FAIL TO STRT", "COOLER #1 VIBR", "COOLER #2 VIBR", "COOLER #3 VIBR", "EICS SD",
     "EICS SHUTDOWN", "EMERGENCY SD", "ENG OVERSPEED", "ENG PANEL SD", "ENG UNDERSPEED", "ENGINE OVERSPD",
     "ENGINE SHUTDOWN", "ENGINE UNDERSPD", "FAILURE TO CRANK", "HI  SUCT PRS", "HI 2 STG SCB LVL", "HI CMP VIB",
     "HI COMP OIL TMP", "HI COMPOR VIB", "HI COMPRESS  OIL", "HI COOLER VIB", "HI COOLER VIBR", "HI CYL2 DISC TP",
     "HI DISC PRESS", "HI DISC PRS", "HI DISC TP CYL 1", "HI DISCH CYL 2 T", "HI DISCH CYL 4 T", "HI DISCHARG  PRS",
     "HI ENG OIL TMP", "HI ENG VIBR", "HI ENG WTR TEMP", "HI ENGINE VIB", "HI ENGINE VIBR", "HI FUEL SCRB LVL",
     "HI FUEL SCRUB  L", "HI INTERSTG1  PR", "HI INTERSTG2 PRS", "HI STAGE1 PRS", "HI STAGE2 PRS", "HI STG1 DISC PRS",
     "HI STG1 DISC TMP", "HI STG1 SCBR LVL", "HI STG1 SCRB LVL", "HI STG1 SCRUB  L", "HI STG1 SUCT PRS",
     "HI STG2 DIS PRS", "HI STG2 DISC TMP", "HI STG2 SCBR LVL", "HI STG2 SCRB LVL", "HI STG2 SCRUB  L",
     "HI STG3 DIS PRS", "HI STG3 DISC IMP", "HI STG3 DISC PRS", "HI STG3 SCBR LVL", "HI SUCT PRESS", "HI SUCT SCRB LVL",
     "HI SUCT SCRUB  L", "HI SUCT SD", "HI SUCTION PRS", "HI SUCTION TMP", "HI TANK LVL", "LB LUBE NO FLOW",
     "LB LUBE NOFLOW", "LO 1ST STG PRESS", "LO 2ND STG PRESS", "LO AUX WATER LVL", "LO AUX WTR LVL", "LO CMP OIL PRESS",
     "LO COMP OIL LVL", "LO COMP OIL PRS", "LO COMPRESS OIL", "LO DISC PRESS", "LO DISC PRS", "LO DISCHARG PRS",
     "LO ENG COOL LVL", "LO ENG JACKETWTR", "LO ENG OIL LVL", "LO ENG OIL PRESS", "LO ENG WTR LVL", "LO INTERSTG1 PRS",
     "LO INTERSTG2 PRS", "LO STAGE1 PRS", "LO STAGE2 PRS", "LO STG1 DIS PRS", "LO STG1 DISC PRS", "LO STG2 DIS PRS",
     "LO STG2 DISC PRS", "LO STG3 DIS PRS", "LO STG3 DISC PRS", "LO SUC PRS SD", "LO SUCT PRESS", "LO SUCT PRS",
     "LO SUCTION PRS", "LOSS OF RPM", "LOST CMP OIL XMT", "LOST COMP OIL PR", "LOST DISC XMTR", "LOST RPM SIGNAL",
     "LOST RPM/STALL", "LOST STG2 PRS XM", "LOST STG2 XMTR", "LUBE NO FLOW", "OVERSPEED", "PANEL ESD",
     "RB LUBE NO FLOW", "SPARE D117 SD", "SPARE DI-03 SD", "SPARE DI-13 SD", "SPARE SHUTDOWN", "UNDERSPEED",
     "UNEXPECTED START"])


def convert_sd_status_code(x):
    if x and str(x).strip().upper() in failure_status:
        return 1
    else:
        return 0


def convert_sd_run_status(x):
    # print("*",x)
    if x and str(x).strip() not in ["1-RUNNING", "2-LOCAL COMM FAI", "2-LOCAL COMM FAIL"]:
        return 1
    else:
        return 0


df['is_failed'] = df["SD Status Code"].apply(convert_sd_status_code) | df["Run Status"].apply(convert_sd_run_status)
df_validation['is_failed'] = df_validation["SD Status Code"].apply(convert_sd_status_code) | df_validation[
    "Run Status"].apply(convert_sd_run_status)

all_feature_cols = PRESSURE_INDICATORS
all_feature_cols.extend(TEMPERATURE_INDICATORS)
all_feature_cols.remove("Cylinder 3 Discharge Temp")
all_feature_cols.remove("Engine Oil Pressure")
all_feature_cols.remove("Compressor Oil Pressure")
all_feature_cols.extend(["Gas Flow Rate", "Gas Flow Rate_RAW", "RPM"])

target_column = ["is_failed"]

print("SCALING")
df = df.fillna(0)
scaler = preprocessing.StandardScaler().fit(df[all_feature_cols])

print("TRAIN TEST SPLITTING")
X_train = scaler.transform(df[all_feature_cols])
y_train = df[target_column].values

X_test = scaler.transform(df_validation[all_feature_cols])
y_test = df_validation[target_column].values

print("TRAININING")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, n_jobs=8, verbose=1)
clf.fit(X_train, y_train)

print("PREDICTING")
prediction = clf.predict(X_test)

total_count = 0
true_positive_count = 0

false_positive = 0
for entry1, entry2 in zip(y_test.reshape(prediction.shape), prediction):
    if entry1 == 1:
        total_count += 1
        if (entry1 == entry2):
            true_positive_count += 1
    elif entry1 == 0 and entry2 == 1:
        false_positive += 1

print("TruePositive: %s, FalsePositive: %s, Total: %s" % ( true_positive_count, false_positive, total_count))

df_validation["prediction"] = prediction

selected_df = df_validation[df_validation["prediction"] == 1]
print("Times and dates when a failure was forecasted")
print(selected_df[["Timestamp", "Facility ID", "prediction"]])

joblib.dump(clf, model_name)

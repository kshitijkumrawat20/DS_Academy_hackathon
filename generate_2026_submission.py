import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. Load 2021 Data to Train the Model
data_dir = 'data'
files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$') and 'Template' not in f]

dfs = []
for file in files:
    file_path = os.path.join(data_dir, file)
    print(f"Reading training data from: {file}...")
    df_temp = pd.read_excel(file_path, header=None)
    
    # Simple header search mechanism
    try:
        header_idx = df_temp[df_temp[0] == 'STATE/UT NAME'].index[0]
    except:
        header_idx = 2 # Best guess fallback
        
    df = pd.read_excel(file_path, header=header_idx)
    df = df.rename(columns={
        'STATE/UT NAME': 'State',
        'AC NO.': 'Constituency_No',
        'AC NAME': 'Constituency_Name',
        'CANDIDATE NAME': 'Candidate Name',
        'PARTY': 'Party',
        'TOTAL': 'Total_Votes'
    })
    
    if 'Constituency_Name' not in df.columns: continue
    
    df = df.dropna(subset=['Candidate Name', 'Constituency_Name', 'Total_Votes'])
    df = df[df['Candidate Name'] != 'NOTA']
    
    # True Winners in 2021
    df['Total_Votes'] = pd.to_numeric(df['Total_Votes'], errors='coerce').fillna(0)
    df['Win'] = df.groupby('Constituency_No')['Total_Votes'].transform(lambda x: (x == x.max()).astype(int))
    
    df['State'] = df['State'].astype(str).str.title().str.strip()
    df = df[['State', 'Constituency_Name', 'Party', 'Win']]
    dfs.append(df)

train_df = pd.concat(dfs, ignore_index=True)

# Engineer Train Features
major_parties = ['BJP', 'INC', 'AITC', 'DMK', 'AIADMK', 'CPI(M)', 'JD(U)', 'AGP', 'AIUDF', 'TMC', 'IUML']
train_df['is_major_party'] = train_df['Party'].apply(lambda x: 1 if any(p in str(x).upper() for p in major_parties) else 0)

# Label Encoding globally across Both Train & Test
le_party = LabelEncoder()
le_const = LabelEncoder()
le_state = LabelEncoder()

# To handle unknown parties/constituencies in the 2026 template safely
class SafeEncoder:
    def __init__(self, le):
        self.le = le
    def fit(self, data):
        self.le.fit(list(data) + ['UNKNOWN'])
    def transform(self, data):
        data = [x if x in self.le.classes_ else 'UNKNOWN' for x in data]
        return self.le.transform(data)

safe_party = SafeEncoder(le_party)
safe_const = SafeEncoder(le_const)
safe_state = SafeEncoder(le_state)

# We fit encoders on the COMBINED vocabulary to avoid issues in 2026 data
# First load test data to gather all names
template_file = 'data/Submission Template.xlsx'
sheets = ['Assam', 'Kerala', 'Puducherry', 'Tamil Nadu', 'West Bengal']

all_parties = set(train_df['Party'].dropna().astype(str))
all_consts = set(train_df['Constituency_Name'].dropna().astype(str))
all_states = set(train_df['State'].dropna().astype(str))

test_dict = {}
for sheet in sheets:
    test_df = pd.read_excel(template_file, sheet_name=sheet)
    test_dict[sheet] = test_df
    all_parties.update(test_df['Party'].dropna().astype(str))
    all_consts.update(test_df['Constituency'].dropna().astype(str))
    all_states.update(test_df['State/UT'].dropna().astype(str))

safe_party.fit(all_parties)
safe_const.fit(all_consts)
safe_state.fit(all_states)

train_df['Party_encoded'] = safe_party.transform(train_df['Party'].astype(str))
train_df['Constituency_Name_encoded'] = safe_const.transform(train_df['Constituency_Name'].astype(str))
train_df['State_encoded'] = safe_state.transform(train_df['State'].astype(str))

features = ['Party_encoded', 'Constituency_Name_encoded', 'State_encoded', 'is_major_party']
X_train = train_df[features]
y_train = train_df['Win']

# Train Final Model on 100% of 2021 Data
print("\nTraining XGBoost on 2021 ECI Data...")
model = XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.8, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 2. Predict on 2026 Template
print("\nGenerating 2026 Predictions from Template...")
output_dir = 'State_Predictions'
os.makedirs(output_dir, exist_ok=True)

for sheet in sheets:
    test_df = test_dict[sheet].copy()
    
    # Process Test data
    test_df['is_major_party'] = test_df['Party'].apply(lambda x: 1 if any(p in str(x).upper() for p in major_parties) else 0)
    
    # Protect encodings
    X_test = pd.DataFrame()
    X_test['Party_encoded'] = safe_party.transform(test_df['Party'].astype(str))
    
    # Use fallback if constituency column naming varies
    const_col = 'Constituency' if 'Constituency' in test_df.columns else 'Constituency_Name'
    X_test['Constituency_Name_encoded'] = safe_const.transform(test_df[const_col].astype(str))
    
    state_col = 'State/UT' if 'State/UT' in test_df.columns else 'State'
    X_test['State_encoded'] = safe_state.transform(test_df[state_col].astype(str))
    X_test['is_major_party'] = test_df['is_major_party']
    
    # Predict Probabilities rather than raw classes to enforce exactly 1 Winner per Constituency
    test_df['WinTarget_Prob'] = model.predict_proba(X_test)[:, 1]
    
    # Enforce exact bounds
    test_df['Predicted Outcome (W/L/O)'] = 'L' # default
    
    # Group by Constituency and pick max prob
    winners_idx = test_df.groupby(const_col)['WinTarget_Prob'].idxmax()
    test_df.loc[winners_idx, 'Predicted Outcome (W/L/O)'] = 'W'
    
    # Drop temporary col
    test_df = test_df.drop(columns=['is_major_party', 'WinTarget_Prob'])
    
    # Save to individual Excel files in the folder
    out_path = os.path.join(output_dir, f"{sheet}.xlsx")
    test_df.to_excel(out_path, index=False)
    print(f"Predicted Outcomes evaluated for -> {sheet} | Saved to: {out_path}")

print(f"\nSUCCESS! Created individual state prediction files in the '{output_dir}' folder.")
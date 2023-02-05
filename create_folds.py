import argparse
import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("--inp_file", default="../Input/raw/train_labels.csv", type=str)
parser.add_argument("--n_folds", default=5, type=int)
parser.add_argument("--target", default="target-label", type=str)
parser.add_argument("--out_file", default=f"../Input/prep/exp1/train_fold-5.csv", type=str)
args = parser.parse_args()


train = pd.read_csv(args.inp_file)

skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=518)
#oof = []
#targets = []
target = args.target

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train, train[target])
):
    train.loc[val_idx, "fold"] = int(fold)



train.to_csv(args.out_file, index=False)

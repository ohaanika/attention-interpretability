import pandas as pd

DATASET = 'IMDB'
DATASPLITS = ['train', 'valid', 'test']
DATACOLS = ["sentence", "label"]

def get_data():
    """
    Read files respective to desired dataset.
    :return: dictionary of datasets, where each dataset is a pandas dataframe.
    """
    data = {}
    for split in DATASPLITS:
        filename = "../../data/"+DATASET+"-"+split+".txt"
        data[split] = pd.read_csv(filename, sep="\t", header=None, names=DATACOLS)
        # with open(filename, "r") as f:
        #     data[split] = f.read().splitlines()
    return(data)

def pickle_data(data):
    """
    Convert pandas dataframes into pickled files and save at in folder containing data.
    :param data: dictionary of datasets, where each dataset is a pandas dataframe.
    """
    for split in DATASPLITS:
        filename = "../../data/"+DATASET+"-"+split+".pkl"
        pd.to_pickle(data[split], filename)

data = get_data()
print(data['train'].head())
pickle_data(data)
                            ### Convert XML format multi-label files into CSV file ###

# Import Libraries
import xml.etree.ElementTree as ET
import glob
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def dl(path):
    g = dict()
    files = glob.glob(path + "/*.xml")
    for file in files:
        print(file)
        tree = ET.parse(file)
        root = tree.getroot()
        b = []
        for image_name in root.iter('filename'):
            image_name = image_name.text
        for member in root.findall('object'):
            a = (member[0].text)
            b.append(a)

        g[image_name] = b

    # Convert Data into DataFrame
    column = ["Images", 'labels']
    xml = pd.DataFrame(g.items(),columns=column,  index=None)
    mlb = MultiLabelBinarizer(sparse_output=True)
    xml = xml.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(xml.pop('labels')),
            columns=mlb.classes_))

    xml.to_csv(path + "_file.csv", index=None)
    return xml

path = "E:/GitHub/pytorch_projects/multi-label_classification/data/Labels"
dl(path)


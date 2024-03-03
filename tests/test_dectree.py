from mlem.dataset import Dataset
from mlem.dectree import ID3Classifier

def main():
    dataset = Dataset.from_csv("datasets/tennis.csv")
    dataset.remove_attr("day") # remove non-data attributes

    clf = ID3Classifier().fit(dataset)
    print(clf.tree_to_dot("tennis"))

if __name__ == "__main__":
    main()


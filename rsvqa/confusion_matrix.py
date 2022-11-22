import argparse
import pandas as pd
import numpy as np

import json



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Evaluate")

    parser.add_argument('--results_file', type = str, required = False, default = "test-results/HR-fullarch-2.pt_preds.csv", help = "path for results")
    parser.add_argument('--save_dir', type = str, required = False, default = "test-results/confusion_matrix/", help = "path to save weights")
    parser.add_argument('--dataset', type = str, required = True, default = "lr", choices=['lr','hr','xben'], help = "dataset to be studied")

    args = parser.parse_args()

    ds = ['lr','hr','xben']
    dirs = {d: args.save_dir + d + '.json' for d in ds}
    
    results = pd.read_csv(args.results_file)

    dataset = args.dataset
    print('dataset', dataset)
    file_path = dirs[dataset]

    # load json with desired classes for confusion matrix
    # json with structure {'idx': <list of integer ids>, 'labels': <corresponding label names>}
    with open(file_path) as f:
        data = json.load(f)

    idx = data['idx']
    label_names = data['labels']

    from sklearn.metrics import confusion_matrix

    def cm():
        mat = confusion_matrix(results["answer"], results["preds"], labels=idx)
        print(mat)

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib as mpl

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(np.log(mat+1), cmap="YlGn")
        
        colorbar_ticks = np.array([0,2,4,6,8,10,12,14])
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels(['{:.0f}'.format(x) for x in np.exp(colorbar_ticks)-1])
        
        ax.set_xticklabels([''] + label_names, rotation = 45, ha="left") #[mapping[l] for l in labels]
        ax.set_yticklabels([''] + label_names) #[mapping[l] for l in labels]
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.tight_layout()
        fig.savefig(args.save_dir+'confusion_matrix_' + dataset + "_" + str(len(idx)) + '.svg')
        plt.show()
        
    
    cm()
import os
import subprocess
from tqdm import tqdm

# for scene in os.listdir(data_folder):s

def WriteFilelist(folder, savepath):

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    rel_folder = os.path.sep.join(folder.split(os.path.sep)[1:])
    with open(os.path.join(savepath, os.path.basename(folder)+'.lst'), 'w+') as list:
        for patch in tqdm(os.listdir(folder), desc=f'Writing scene {os.path.basename(folder)} file list...'):
            if patch.endswith('.xml'):
                continue
            list.write('./'+os.path.join(rel_folder, patch).replace(os.path.sep, '/')+'\n')
        list.close()

def WriteDataset(folder, savepath):

    for scene in os.listdir(folder):
        WriteFilelist(os.path.join(folder, scene), savepath)

def ScenePrediction(listpath):
    """data\list\validacion\test"""
    command_ = [
        'python', r'tools\test.py',
        '--cfg', 'experiments\customdataset\config.yaml',
        'TEST.PREDICT_MODE', 'test',
        'TEST.BATCH_LISTDIR', listpath.replace(os.path.sep, '/')
    ]
    subprocess.run(command_, shell=True)

def main():
    data_folder = os.path.normpath('./data/validacion')
    listpath = os.path.normpath('./data/list/validacion')

    WriteDataset(data_folder, listpath)
    # ScenePrediction(listpath)

if __name__ == '__main__':
    main()
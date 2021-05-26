import os
import glob
from shutil import copyfile

root_dir = '/home/shuki/HarbThesis/HarbThesis/New_6Classes/Uhren_6'
split_file = '/home/shuki/HarbThesis/HarbThesis/New_6Classes/Uhren_6_split/split5.txt'
train_dir = '/home/shuki/Documents/Experiments/ClockTestCrossValidation/6Classes/Split5/Data/train/6/'
valid_dir = '/home/shuki/Documents/Experiments/ClockTestCrossValidation/6Classes/Split5/Data/valid/6/'
test_dir = '/home/shuki/Documents/Experiments/ClockTestCrossValidation/6Classes/Split5/Data/test/6/'

def readSplitFile():
    with open(split_file) as f:
        trainings = []
        valids = []
        tests = []
        # -1: none, 0: trainings, 1: valids, 2:tests
        goal_dataset = -1
        for line in f:
            if 'training dataset' in line:
                goal_dataset = 0
            elif 'valiation dataset' in line:
                goal_dataset = 1
            elif 'test dataset' in line:
                goal_dataset = 2
            else:
                if goal_dataset == 0:
                    trainings = line.split(";")
                elif goal_dataset == 1:
                    valids = line.split(";")
                elif goal_dataset == 2:
                    tests = line.split(";")
                else:
                    continue
        return trainings, valids, tests


def searchFile(keyword, search_path):
    # Append a directory separator if not already present
    if not (search_path.endswith("/") or search_path.endswith("\\")):
        search_path = search_path + "/"

    # If path does not exist, set search path to current directory
    if not os.path.exists(search_path):
        search_path = "."

    for filename in glob.iglob(os.path.join(search_path, '*'+ keyword + '*'), recursive=False):
        return os.path.basename(filename)

    #default
    return ""


def copyFiles(namelist, root_dir, goal_dir):
    if not os.path.exists(goal_dir):
        os.mkdir(goal_dir)

    for name in namelist:
        filename = searchFile(name, root_dir)
        if filename == "":
            continue

        srcname = os.path.join(root_dir, filename)
        dstname = os.path.join(goal_dir, filename)
        if os.path.exists(srcname):
            copyfile(srcname,dstname)


def prepareData():
    train_ds, valid_ds, test_ds = readSplitFile()
    copyFiles(train_ds, root_dir, train_dir)
    copyFiles(valid_ds, root_dir, valid_dir)
    copyFiles(test_ds, root_dir, test_dir)


if __name__ == "__main__":
    prepareData()
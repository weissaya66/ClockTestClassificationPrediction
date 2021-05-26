from enum import Enum

class ModelType(Enum):
    VGG16=1
    ResNet152=2
    DenseNet121=3


# Data directory
test_dir = '/home/shuki/Documents/Experiments/ClockTest/6Classes/Data/test/'
output_dir = '/home/shuki/Documents/Experiments/ClockTest/6Classes/'
output_best_checkpoit = "/home/shuki/Documents/ClockTest/CodePred/models/6ClassesCheckPoint.pth"

# specify the image classes
classes = ['1','2','3' ,'4' ,'5' ,'6'] #for 6 classes
#classes = ['pass', 'fail'] #for 2 classes
n_class = len(classes)

# model type
model_type = ModelType.DenseNet121

# hyper parameters
#learning_rate = 0.0001
#step_size = 7
batch_size = 16

# prediction
output_results = output_dir + "/resultTmp.txt"



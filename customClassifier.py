from torch import nn
import config

class ClocktestClassifier(nn.Module):
    def __init__(self):
        super(ClocktestClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, config.n_class)
        # self.fc2 = nn.Linear(800, 800)
        # self.fc3 = nn.Linear(800, 400)
        # self.fc4 = nn.Linear(400, 6)
        # self.fc5 = nn.Linear(500, 6)

        # dropout layer (p=0.25)
        #self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        # x = self.dropout(F.relu(self.fc4(x)))

        # output so no dropout here
        x = self.fc1(x)

        return x
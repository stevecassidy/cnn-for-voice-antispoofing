
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))  # 2x4

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 3x3 - 32
        init.xavier_normal_(self.conv1.weight)

        self.mfm1 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))
        self.conv2a = nn.Conv2d(16, 32, kernel_size=1)
        init.xavier_normal_(self.conv2a.weight)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3)  # 3x3 - 32
        init.xavier_normal_(self.conv2.weight)

        self.conv3a = nn.Conv2d(24, 48, kernel_size=1)
        init.xavier_normal_(self.conv3a.weight)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3)  # 3x3 - 32
        init.xavier_normal_(self.conv3.weight)
        self.conv4a = nn.Conv2d(32, 64, kernel_size=1)  # 3x3 - 32
        init.xavier_normal_(self.conv4a.weight)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)  # 3x3 - 32
        init.xavier_normal_(self.conv4.weight)
        self.conv5a = nn.Conv2d(16, 32, kernel_size=3)  # 3x3 - 32
        init.xavier_normal_(self.conv5a.weight)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)  # 3x3 - 32
        init.xavier_normal_(self.conv5.weight)

        self.fc1 = nn.Linear(1512, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # try pooling after relu!
        x = self.pool(self.mfm1(self.conv1(x)))
        x = self.mfm1(self.conv2a(x))
        x = self.pool(self.mfm1(self.conv2(x)))
        x = self.mfm1(self.conv3a(x))
        x = self.pool(self.mfm1(self.conv3(x)))
        x = self.mfm1(self.conv4a(x))
        x = self.pool(self.mfm1(self.conv4(x)))
        x = self.mfm1(self.conv5a(x))
        x = self.pool(self.mfm1(self.conv5(x)))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv1(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv2a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv2(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv3a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv3(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv4a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv4(x), training=self.training)))
        # x = self.mfm1(F.dropout2d(self.conv5a(x), training=self.training))
        # x = self.pool(self.mfm1(F.dropout2d(self.conv5(x), training=self.training)))

        x = x.view(x.size(0), -1)

        #x = nn.functional.dropout(x, p=0.7, training=self.training)
        x = nn.functional.relu(self.fc1(x))
        #x = nn.functional.dropout(x, p=0.7, training=self.training)
        x = self.fc2(x)

        return nn.functional.log_softmax(x, dim=1) 


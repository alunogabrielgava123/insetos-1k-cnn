import torch
import torch.nn as nn


#Simple CNN
class SimpleCNN(nn.Module):
    
    """
    Calculo de parametros dentro desse rede neural
    
    imagens (224,224, 3) -> (224,224, 32) -> (112,112, 32) -> (112,112, 64) -> (56,56, 64) -> (56,56, 128) -> (28,28, 128) -> (15)
    
    calcula o numero de parametros:
        
        (3*3*3 + 1) * 32 = 896
        (3*3*32 + 1) * 64 = 18496
        (3*3*64 + 1) * 128 = 73856
        (128 * 28 * 28 + 1) * 15 = 47055
        total = 140803
    
    calcula das nw e nh das camadas convulacionas:
        formula:
        (nw + 2p -f )/s + 1 -> (224 + 2 - 3)/1 + 1 = 224 tamanho nw e nh
        (nw + 2p -f )/s + 1 -> (224 + 2 - 3)/1 + 1 = 112 ""
        (nw + 2p -f )/s + 1 -> (224 + 2 - 3)/1 + 1 = 56 ""       
    
    """
    
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Camadas convolutivas e pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Camada densa
        self.fc = nn.Linear(in_features=128 * 28 * 28, out_features=15)
        self.dropout_fc = nn.Dropout(0.5)  # Dropout antes da camada densa

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
      
        x = self.maxpool2(self.relu2(self.conv2(x)))

        x = self.maxpool3(self.relu3(self.conv3(x)))
   
        x = x.view(x.size(0), -1)
        #x = self.dropout_fc(x)  # Aplicar dropout
        x = self.fc(x)
        return x

# Criar uma instância do modelo

if __name__ == '__main__':
    model = SimpleCNN()
    #Sumatty model
    print(model)

        
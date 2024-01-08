import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from loaders import LoaderModelDataSet
from model_s_cnn import SimpleCNN
import numpy as np
from train import fit
import torchvision.models as models



    
if  __name__ == '__main__':
    loder_model =  LoaderModelDataSet('farm_insects')
    ( treinamento_vetor, val_loader , teste_loader  ) = loder_model.load(transform=transforms.Compose([
            transforms.RandomResizedCrop(224),        # Cortes aleatórios na imagem e redimensionamento para 224x224
            transforms.RandomHorizontalFlip(),        # Inversão horizontal aleatória
            transforms.RandomRotation(30),            # Rotação aleatória dentro de um intervalo de 30 graus
            transforms.ToTensor()]))

    #definindo o modelo
    model_1 = SimpleCNN(num_classes=15)
    model_2 = SimpleCNN(num_classes=15)
    res_net_18 = models.resnet18(pretrained=True)
    
    #definindo o dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #definir o otimizador
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.001)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.001, momentum=0.9)
    
    #definir a função de perda
    loss_fn = torch.nn.CrossEntropyLoss()
    
    #definir o dataloader
    train_loader = DataLoader(treinamento_vetor, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=32, shuffle=True) 
    
    # #definir o número de épocas
    num_epo = 30
    
    # #treinando modelo com funcoes de treinamento diferentes
    # print("Inicio do treinamento do models")
    # mode_1, history_1  = fit(train_loader, model_1, device , optimizer_1, num_epo, loss_fn,  debug = True, validation_loader=val_loader)
    # print('Treinamenrto do modelo 1 terminado')
    # mode_2, history_2  = fit(train_loader, model_2, device , optimizer_2, num_epo, loss_fn,  debug = True, validation_loader=val_loader)
    
    
    #salvar a historua do treinamento para analise
    import pandas as pd
    # pd.DataFrame(history_1).to_csv("analise_simple_cnn1.csv")
    # pd.DataFrame(history_2).to_csv("analise_simple_cnn2.csv")
    
    
    #treinamento com modelos pre treinados
    #treinamento com resnet18
    #definindo o modelo
    res_net_18 = models.resnet18(pretrained=True)
    res_net_18.fc = torch.nn.Linear(in_features=512, out_features=15)
    res_net_18.to(device)
    
    #definir o otimizador
    optimizer_3 = torch.optim.Adam(res_net_18.parameters(), lr=0.001)
    
    #treinando o modelo
    print("Inicio do treinamento do models")
    mode_3, history_3  = fit(train_loader, res_net_18, device , optimizer_3, num_epo, loss_fn,  debug = True, validation_loader=val_loader)
    print('Treinamenrto do modelo 1 terminado')
    
    #salvar a historua do treinamento para analise
    pd.DataFrame(history_3).to_csv("analise_resnet18.csv")
    
    
    
    #salvando o modelo 
    #torch.save(mode.state_dict(), 'simple_cnn.pt')
    
    

    
    
    
    
    
    
    
















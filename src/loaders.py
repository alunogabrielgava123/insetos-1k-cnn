from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class LoaderModelDataSet:
    def __init__(self, file_name: str, trainin_size = 0.7, teste_size = 0.15, validation = 0.15):
        self.file_name = file_name
        self.trainin_size = trainin_size
        self.teste_size = teste_size
        self.validation = validation

    def load(self, transform):
        #verificando se o transformer foi passado casso controle soltar um erro usando asert
        assert transform is not None, "transformer não pode ser None, passe um valor trasformador.compose"
        
        dataset = datasets.ImageFolder(self.file_name, transform=transform)
        
        tamanho_total = len(dataset)
        tamanho_treino = int(self.trainin_size * tamanho_total)
        tamanho_teste = int(self.teste_size * tamanho_total)
        tamanho_validacao = tamanho_total - tamanho_treino - tamanho_teste

        conjunto_treino, conjunto_validacao, conjunto_teste = random_split(dataset, [tamanho_treino, tamanho_validacao, tamanho_teste])

        return conjunto_treino, conjunto_validacao, conjunto_teste
        


#script de teste
if __name__ == '__main__':
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Loader_model
    loader_model = LoaderModelDataSet('../farm_insects', 0.7, 0.15, 0.15)
    ( conjuento_treinamento, conjunto_teste, conjuento_validacao ) = loader_model.load(transform=transform)
    print(f'Conjunto de treinamento: {len(conjuento_treinamento)}, Conjunto de teste: {len(conjunto_teste)}, Conjunto de validação: {len(conjuento_validacao)}')
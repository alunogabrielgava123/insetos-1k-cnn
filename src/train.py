import torch

def fit(train_loader, model, device, optimizer, num_epo, loss_fn, debug=False, validation_loader=None):
    
    history = {'loss_traning': [], 'loss_validation': [], 'acuracia_val': [], 'acuracia_traning': []}
    
    model.to(device)

    for epoch in range(num_epo):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, t_predict = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (t_predict == labels).sum().item()

        epoch_loss_train = running_loss / len(train_loader)
        epoch_acc_train = 100 * correct_train / total_train
        history['loss_traning'].append(epoch_loss_train)
        history['acuracia_traning'].append(epoch_acc_train)

        if validation_loader is not None:
            model.eval()
            val_loss = 0.0
            total_val = 0
            correct_val = 0

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, val_predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (val_predicted == labels).sum().item()

            epoch_loss_val = val_loss / len(validation_loader)
            epoch_acc_val = 100 * correct_val / total_val
            history['loss_validation'].append(epoch_loss_val)
            history['acuracia_val'].append(epoch_acc_val)

            if debug:
                print(f'Época [{epoch + 1}/{num_epo}], Loss Treinamento: {epoch_loss_train:.4f}, Acurácia Treinamento {epoch_acc_train:.4f}%  |  Loss Validação: {epoch_loss_val:.4f}, Acurácia Validação: {epoch_acc_val:.2f}%')

    return model, history
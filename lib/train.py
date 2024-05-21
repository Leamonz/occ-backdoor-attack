from tqdm import tqdm
import torch
import os


def simple_train(model, train_dataloader, num_epochs, criterion, optimizer, val_dataloader, device, model_save_dir):
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        train_accs = []
        for batch in tqdm(train_dataloader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        valid_loss = []
        valid_accs = []
        for batch in tqdm(val_dataloader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
                labels = labels.to(device)
            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[Valid | {epoch + 1:03d}/{num_epochs:03d}] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(model_save_dir, str(acc)))
            print("saving model with acc {:.3f}".format(best_acc))

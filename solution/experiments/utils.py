
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from IPython.display import clear_output


def evaluate_metric(clf, dataloader, evaluate_on_start):
    clf.eval()
    correct_predictions = 0
    for text_batch, label_batch, start_batch, end_batch in dataloader:
        with torch.no_grad():
            y_pred = clf(text_batch).detach().numpy()
        predicted_labels = np.eye(y_pred.shape[1])[np.argmax(y_pred, axis=1)]

        if evaluate_on_start:
            batch_correct_predictions = np.all(start_batch.numpy() == predicted_labels, axis=1).sum()
        else:
            batch_correct_predictions = np.all(end_batch.numpy() == predicted_labels, axis=1).sum()
        correct_predictions += batch_correct_predictions

    return correct_predictions / len(dataloader.dataset)


def evaluate_metrics(clf, train_dataset, test_dataset, evaluate_on_start=True):
    if test_dataset:
        test_metric = evaluate_metric(clf, test_dataset, evaluate_on_start)
    else:
        test_metric = None

    return evaluate_metric(clf, train_dataset, evaluate_on_start), test_metric


def show_training(losses, train_scores, test_scores):
    clear_output(True)

    plt.figure(figsize=[16, 9])

    plt.subplot(2, 2, 1)
    plt.title("Loss")
    plt.plot(losses, label='Cross Entropy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_scores, label='train')
    plt.plot(test_scores, label='test')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_loop(
        n_epoch,
        clf,
        train_dataloader,
        test_dataloader=None,
        lr=0.001,
        train_on_start=True
        ):
    optimizer = torch.optim.Adam(
        params=clf.parameters(),
        lr=lr
    )
    loss = nn.CrossEntropyLoss()

    losses = []
    train_scores = []
    test_scores = []

    for epoch in range(n_epoch):
        clf.train()
        batch_losses = []
        for text_batch, label_batch, start_batch, end_batch in train_dataloader:
            optimizer.zero_grad()

            y_pred = clf(text_batch)
            if train_on_start:
                output = loss(y_pred, start_batch)
            else:
                output = loss(y_pred, end_batch)

            batch_losses.append(output.item())

            output.backward()
            optimizer.step()

        epoch_train_acc, epoch_test_acc = evaluate_metrics(clf, train_dataloader, test_dataloader, train_on_start)
        losses.append(np.mean(batch_losses))
        train_scores.append(epoch_train_acc)
        test_scores.append(epoch_test_acc)

        show_training(losses, train_scores, test_scores)

import os
import pickle
import torch
import json
from scripts.utils import load_vocab, read_corpus
from scripts.config import Config
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from model.BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from scripts.utils import train, valid
import torchsnooper

# @torchsnooper.snoop()
def main():
    device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    res = []  # 用来保存输出的路径结果（数字表示路径）
    print('loading corpus')
    config = Config()
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tags_name = []
    tags_name.extend(label_dic) #把标签名字存入列表中
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

    #test
    test_ids = torch.LongTensor([temp.input_id for temp in test_data])
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
    test_tags = torch.LongTensor([temp.label_id for temp in test_data])

    test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    model = BERT_BiLSTM_CRF(tagset_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            config.filters_number,
                            # config.num_gcn_layers,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)

    # 验证集最好结果
    best_score = 0.0
    best_accuracy = 0.0
    best_recall = 0.0
    epoch_dev = 0.0
    start_epoch = 1
    # 测试集最好结果
    best_score_test = 0.0
    best_accuracy_test = 0.0
    best_recall_test = 0.0
    epoch_test = 0.0
    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, start_estimator,_ = valid(model,
                                           dev_loader)
    print("\t* Validation loss in dev before training : loss = {:.4f}, accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
          .format(valid_loss, (start_estimator[0] * 100), (start_estimator[1] * 100), (start_estimator[2] * 100)))

    _, valid_loss, start_estimator,_ = valid(model,
                                           test_loader)
    print("\t* Validation loss in test before training : loss = {:.4f}, accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
          .format(valid_loss, (start_estimator[0] * 100), (start_estimator[1] * 100), (start_estimator[2] * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training BERT_BiLSTM_CRF model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, config.epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss = train(model,
                                       train_loader,
                                       optimizer,
                                       config.max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}"
              .format(epoch_time, epoch_loss))
        #dev result
        epoch_time, valid_loss, valid_estimator,valid_estimator_1 = valid(model,
                                                        dev_loader)
        valid_losses.append(valid_losses)
        print(20 * "-", "dev result", 20 * "-")
        #输出每个实体的P R F1
        i =1
        for num in range(1,len(tags_name)-3,2):
            print("tag name:",tags_name[num]," -> accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][i] * 100),
                                                                                                          (valid_estimator_1[1][i] * 100),
                                                                                                          (valid_estimator_1[2][i] * 100)))
            i += 1
        # print(" -> tag name :PER,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][1] * 100),(valid_estimator_1[1][1] * 100),(valid_estimator_1[2][1] * 100)),
        #       "\n","-> tag name :ECO,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][2] * 100),(valid_estimator_1[1][2] * 100),(valid_estimator_1[2][2] * 100)),
        #       "\n","-> tag name :MAG,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][3] * 100),(valid_estimator_1[1][3] * 100),(valid_estimator_1[2][3] * 100)),
        #       "\n","-> tag name :PTN,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][4] * 100),(valid_estimator_1[1][4] * 100),(valid_estimator_1[2][4] * 100)),
        #       "\n","-> tag name :MAK,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][5] * 100),(valid_estimator_1[1][5] * 100),(valid_estimator_1[2][5] * 100)),
        #       "\n","-> tag name :PLR,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][6] * 100),(valid_estimator_1[1][6] * 100),(valid_estimator_1[2][6] * 100)),
        #       "\n","-> tag name :ETF,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][7] * 100),(valid_estimator_1[1][7] * 100),(valid_estimator_1[2][7] * 100)),
        #       "\n","-> tag name :PPR,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][8] * 100),(valid_estimator_1[1][8] * 100),(valid_estimator_1[2][8] * 100)),
        #       "\n","-> tag name :INS,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][9] * 100),(valid_estimator_1[1][9] * 100),(valid_estimator_1[2][9] * 100)),)
        print("-> total result : Valid time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
              .format(epoch_time, valid_loss, (valid_estimator[0] * 100), (valid_estimator[1] * 100), (valid_estimator[2] * 100)))

        #test result
        epoch_time_test, valid_loss_test, valid_estimator_test,valid_estimator_1 = valid(model,
                                                        test_loader)
        print(20 * "-", "test result", 20 * "-")
        # 输出每个实体的P R F1
        j = 1
        for num in range(1, len(tags_name) - 3, 2):
            print("tag name:", tags_name[num],
                  " -> accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][j] * 100),
                                                                               (valid_estimator_1[1][j] * 100),
                                                                               (valid_estimator_1[2][j] * 100)))
            j += 1
        # print(" -> tag name :PER,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][1] * 100),(valid_estimator_1[1][1] * 100),(valid_estimator_1[2][1] * 100)),
        #       "\n","-> tag name :ECO,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][2] * 100),(valid_estimator_1[1][2] * 100),(valid_estimator_1[2][2] * 100)),
        #       "\n","-> tag name :MAG,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][3] * 100),(valid_estimator_1[1][3] * 100),(valid_estimator_1[2][3] * 100)),
        #       "\n","-> tag name :PTN,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][4] * 100),(valid_estimator_1[1][4] * 100),(valid_estimator_1[2][4] * 100)),
        #       "\n","-> tag name :MAK,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][5] * 100),(valid_estimator_1[1][5] * 100),(valid_estimator_1[2][5] * 100)),
        #       "\n","-> tag name :PLR,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][6] * 100),(valid_estimator_1[1][6] * 100),(valid_estimator_1[2][6] * 100)),
        #       "\n","-> tag name :ETF,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][7] * 100),(valid_estimator_1[1][7] * 100),(valid_estimator_1[2][7] * 100)),
        #       "\n","-> tag name :PPR,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][8] * 100),(valid_estimator_1[1][8] * 100),(valid_estimator_1[2][8] * 100)),
        #       "\n","-> tag name :INS,accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format((valid_estimator_1[0][9] * 100),(valid_estimator_1[1][9] * 100),(valid_estimator_1[2][9] * 100)),)
        print("-> total result : Valid time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
              .format(epoch_time_test, valid_loss_test, (valid_estimator_test[0] * 100), (valid_estimator_test[1] * 100), (valid_estimator_test[2] * 100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(valid_estimator[0])

        # 记录测试集最好结果
        if valid_estimator_test[2] > best_score_test:
            best_score_test = valid_estimator_test[2]
            best_accuracy_test = valid_estimator_test[0]
            best_recall_test = valid_estimator_test[1]
            epoch_test = epoch

        # Early stopping on validation accuracy.  estimator[0]: 准确率
        # 记录验证集最好结果
        if valid_estimator[2] < best_score:
            patience_counter += 1
        else:
            best_score = valid_estimator[2]
            best_accuracy = valid_estimator[0]
            best_recall = valid_estimator[1]
            epoch_dev = epoch
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'esim_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(config.target_dir, "RoBERTa_best.pth.tar"))

        # Save the model at each epoch.
        # torch.save({"epoch": epoch,
        #             "model": model.state_dict(),
        #             "best_score": best_score,
        #             "optimizer": optimizer.state_dict(),
        #             "epochs_count": epochs_count,
        #             "train_losses": train_losses,
        #             "valid_losses": valid_losses},
        #            os.path.join(config.target_dir, "RoBERTa_NER_{}.pth.tar".format(epoch)))

        # Early stopping  并且输出测试集和验证集最好的结果
        if patience_counter >= config.patience:
            print(30 * "-", "best dev result", 30 * "-")
            print("-> epoch {} , accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
                  .format(epoch_dev, (best_accuracy * 100),
                          (best_recall * 100),
                          (best_score * 100)))

            print(30 * "-", "best test result", 30 * "-")
            print("-> epoch {} , accuracy: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%"
                  .format(epoch_test, (best_accuracy_test * 100),
                          (best_recall_test * 100),
                          (best_score_test * 100)))

            print("-> Early stopping: patience limit reached, stopping...")
            break
    # Plotting of the loss curves for the train and validation sets.
    # plt.figure()
    # plt.plot(epochs_count, train_losses, "-r")
    # plt.plot(epochs_count, valid_losses, "-b")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend(["Training loss", "Validation loss"])
    # plt.title("Cross entropy loss")
    # plt.show()
    # plt.savefig('../result/loss.png')


if __name__ == "__main__":
    main()






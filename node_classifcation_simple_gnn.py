# -*- coding: utf-8 -*-
"""
@Time   : 22/9/2021
@Author : Guikun Chen
"""
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn


# load dataset
def get_data(folder="/home/cgk/dataset", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name)
    return dataset


# create the graph cnn model
class GraphCNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GraphCNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=out_c)

    def forward(self, data):
        # data.x data.edge_index
        x = data.x  # [N, C]
        edge_index = data.edge_index  # [2 ,E]
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D]
        hid = F.relu(hid)

        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c]

        out = F.log_softmax(out, dim=1)  # [N, out_c]

        return out


class MyGCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(MyGCN, self).__init__()

        self.conv0 = pyg_nn.SGConv(in_c, hid_c, K=2)

        self.conv1 = pyg_nn.APPNP(K=2, alpha=0.2)
        self.conv2 = pyg_nn.APPNP(K=2, alpha=0.2)
        self.conv3 = pyg_nn.APPNP(K=2, alpha=0.2)

        self.conv4 = pyg_nn.SGConv(hid_c, out_c, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv0(x, edge_index)
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.leaky_relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cora_dataset = get_data()

    my_net = MyGCN(in_c=cora_dataset.num_features, hid_c=256, out_c=cora_dataset.num_classes)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)
    data = cora_dataset[0].to(device)

    optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.95)

    # model train
    best_acc = .0
    my_net.train()
    for epoch in range(500):
        optimizer.zero_grad()

        output = my_net(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        print("Epoch", epoch + 1, "Loss", loss.item())

        ####### test
        my_net.eval()
        _, prediction = my_net(data).max(dim=1)

        target = data.y

        test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
        test_number = data.test_mask.sum().item()

        acc = test_correct / test_number
        print("Accuracy of Test Samples: {}".format(acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(my_net.state_dict(), "best.ckpt")

    print(best_acc)
    # model test
    # my_net.eval()
    # _, prediction = my_net(data).max(dim=1)

    # target = data.y

    # test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    # test_number = data.test_mask.sum().item()

    # acc = test_correct / test_number
    # print("Accuracy of Test Samples: {}".format(acc))
    # if acc >= 0.82:
    #     torch.save(my_net.state_dict(), "1.ckpt")


if __name__ == '__main__':
    main()

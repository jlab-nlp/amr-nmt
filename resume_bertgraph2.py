import json
import numpy as np
import networkx as nx
from pyvis.network import Network
from nltk.tokenize import word_tokenize
import re
import torch
from torch_geometric.data import Data

from transformers import BertTokenizer, BertModel

device = torch.device("cuda:0")

# train_dataset = torch.load('./data/bertgraph/train_multi.pt', map_location=device)
# dev_dataset = torch.load('./data/bertgraph/dev_multi.pt', map_location=device)
# test_dataset = torch.load('./data/bertgraph/test_multi.pt', map_location=device)

train_dataset = torch.load('./data/bertgraph/train_multi.pt')
dev_dataset = torch.load('./data/bertgraph/dev_multi.pt')
test_dataset = torch.load('./data/bertgraph/test_multi.pt')

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of dev graphs: {len(dev_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("------------------data load successfully--------------------------")

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv
from transformers import BertModel, BertPreTrainedModel, BertConfig


# model = BertModel.from_pretrained("bert-base-uncased")
class BERTGCN(BertPreTrainedModel):
    def __init__(self, hidden_channels, config):
        super(BERTGCN, self).__init__(config)
        torch.manual_seed(666)
        self.bert = BertModel(config)
        # train_dataset[0].num_features
        # self.conv1 = GCNConv(config.hidden_size, hidden_channels)
        self.lin = Linear(768, 5)
        # self.lin = Linear(hidden_channels, 5)

    def forward(self, node_input_ids, node_attention_mask, edge_index, batch, labels=None):
        # x, edge_index, batch
        # node_input_ids (number_of_nodes, node_length)
        # node_attention_mask (umber_of_nodes, node_length)
        print("!!!!!start forward!!!")

        outputs = self.bert(node_input_ids, node_attention_mask)
        print("!!!!!finish bert!!!")

        pooled_output = outputs[1]  # (number_of_nodes, hidden_size)
        print("!!!!!get bert cls!!!")
        # print("outputs", outputs)
        # print("pooled_output", pooled_output)

        # x = F.relu(self.conv1(pooled_output, edge_index))
        x = pooled_output
        print(x.shape)

        print("!!!!!finish gcn!!!")

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        print("!!!!!get gcn output!!!")
        print("after global mean pool", x.shape)
        print("after global mean pool", x)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        print("after linear", x.shape)
        print("after linear", x)
        if labels is not None:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(x.view(-1, 5), labels.view(-1))
            return x, loss
        return x


for lr in [0.001]:
    print("#########################", lr, "#############################################")
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BERTGCN(hidden_channels=768, config=config)
    model = model.to(device)

    # torch.cuda.set_device(0)
    # device = torch.device("cuda", 0)

    # torch.distributed.init_process_group(backend="nccl")
    # model = torch.nn.parallel.DistributedDataParallel(model)

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train():
        # model.zero_grad()
        model.train()
        loss_all = 0

        for data in train_loader:  # Iterate in batches over the training dataset.
            # print(data.x)
            data.to(device)
            out, loss = model(data.x, data.z, data.edge_index, data.batch,
                              labels=data.y)  # Perform a single forward pass.
            print("label", data.y)  # Compute the loss.

            # loss = criterion(out, data.y)  # Compute the loss.
            # Compute the loss.

            loss = loss.mean()
            loss.backward()  # Derive gradients.
            loss_all += loss.item() * data.num_graphs
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return loss_all / len(train_dataset)


    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = model(data.x, data.z, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    best_dev_acc = 0
    best_test_acc = 0

    for epoch in range(1, 1001):
        train_loss = train()
        train_acc = test(train_loader)
        dev_acc = test(dev_loader)
        test_acc = test(test_loader)
        best_dev_acc = max(best_dev_acc, dev_acc)
        best_test_acc = max(best_test_acc, test_acc)
        print(
            'Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Dev Acc: {:.7f}, Test Acc: {:.7f}, Best Dev Acc: {:.7f}, Best Test Acc: {:.7f}'.format(
                epoch, train_loss, train_acc, dev_acc, test_acc, best_dev_acc, best_test_acc))

import torch
from torch.autograd import Function
from torch import nn


class DeepDecipherOp(Function):
    @staticmethod
    def forward(self, index, pseudo_label, label_lr, original_num):
        # pseudo_label: Nx200
        self.label_lr = label_lr
        self.original_num = original_num
        batch_label = pseudo_label.index_select(0, index)  # copy
        self.save_for_backward(index, pseudo_label, batch_label)

        return batch_label

    @staticmethod
    def backward(self, gradOutput):
        if self.label_lr == 0:
            return None, None, None, None

        index, pseudo_label, batch_label = self.saved_tensors

        # if any(index < self.original_num):
        #     index_not_update = index < self.original_num  # index of original image
        #     gradOutput[index_not_update, :] = 0

        pseudo_label_update = self.label_lr * gradOutput.data

        if self.label_lr != 0:
            batch_label.data.sub_(pseudo_label_update)
            pseudo_label[index, :] = batch_label.data.cpu()

        return None, None, None, None


class DeepDecipher(nn.Module):

    def __init__(self, datasize, class_num, label_lr, original_num):
        super(DeepDecipher, self).__init__()
        self.label_lr = label_lr
        self.pseudo_label = nn.Parameter(torch.zeros(datasize, class_num))
        self.original_num = original_num
        # self.register_buffer('pseudo_label', torch.zeros(datasize, class_num))

    def forward(self, index):
        out = DeepDecipherOp.apply(index, self.pseudo_label, self.label_lr, self.original_num)
        return out

    def init_label(self, label):
        self.pseudo_label.data = torch.Tensor(label)

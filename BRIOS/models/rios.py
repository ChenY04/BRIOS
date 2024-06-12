import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RIOS_H(nn.Module):
    def __init__(self, rnn_hid_size, SEQ_LEN, SELECT_SIZE):
        super(RIOS_H, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.SEQ_LEN = SEQ_LEN
        self.SELECT_SIZE = SELECT_SIZE
        self.build()

    def build(self):
        self.rnn_cell0 = nn.LSTMCell(self.SELECT_SIZE * 2, self.rnn_hid_size)
        self.rnn_cell1 = nn.LSTMCell(self.SELECT_SIZE * 2, self.rnn_hid_size)
        self.temp_decay_h = TemporalDecay(input_size = self.SELECT_SIZE, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_h0 = TemporalDecay(input_size=self.SELECT_SIZE, output_size=self.rnn_hid_size, diag=False)
        self.hist_reg0 = nn.Linear(self.rnn_hid_size, self.SELECT_SIZE)
        # self.feat_reg0 = FeatureRegression(self.SELECT_SIZE * 2)
        # self.weight_combine = nn.Linear(self.SELECT_SIZE * 2, self.SELECT_SIZE)

    # def forward(self, data, f_data, direct):
    def forward(self, data, direct):

        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        bsize = values.size()[0]

        masksout = torch.reshape(masks, (bsize, self.SEQ_LEN, 1))

        evals = values[:, :, 2]
        evals = torch.reshape(evals, (bsize, self.SEQ_LEN, 1))
        eval_masks = data[direct]['eval_masks']
        eval_masks = torch.reshape(eval_masks, (bsize, self.SEQ_LEN, 1))

        h0 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c0 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        h1 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c1 = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
            h1, c1 = h1.cuda(), c1.cuda()

        x_loss0 = 0.0
        imputations0 = []

        for t in range(self.SEQ_LEN):

            x_y = values[:, t, 2]
            m_y = masks[:, t]
            d_y = deltas[:, t, 2]

            x_y = torch.reshape(x_y, (bsize, 1))
            m_y = torch.reshape(m_y, (bsize, 1))
            d_y = torch.reshape(d_y, (bsize, 1))

            gamma_h = self.temp_decay_h(d_y)

            x_h = self.hist_reg0(h0)
            x_c = m_y * x_y + (1 - m_y) * x_h
            x_loss0 = x_loss0 + torch.sum(torch.square(x_y - x_h) * m_y) / (torch.sum(m_y) + 1e-5)

            inputs1 = values[:, t, 0:2]
            h1, c1 = self.rnn_cell1(inputs1, (h1, c1))

            h1 = h1 * gamma_h

            inputs = torch.cat([x_c, m_y], dim=1)
            h0, c0 = self.rnn_cell0(inputs, (h1, c1))

            imputations0.append(x_c.unsqueeze(dim = 1))

        imputations0 = torch.cat(imputations0, dim = 1)

        return {'loss': x_loss0, 'imputations': imputations0, 'eval_masks': eval_masks, 'evals': evals, 'masks':masksout}

class Model(nn.Module):
    def __init__(self, rnn_hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE):
        super(Model, self).__init__()
        self.RIOS_H = RIOS_H(rnn_hid_size, SEQ_LEN, SELECT_SIZE)

    def forward(self, data, direct):
        out2 = self.RIOS_H(data, direct)

        return {'loss': out2['loss'], 'imputations': out2['imputations'], 'eval_masks': out2['eval_masks'], 'evals': out2['evals'], 'masks': out2['masks']}


    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
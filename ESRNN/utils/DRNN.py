# Dilated Recurrent Neural Networks. https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf
# implementation from https://github.com/zalandoresearch/pytorch-dilated-rnn
# Residual LSTM: Design of a Deep Recurrent Architecture for Distant Speech Recognition. https://arxiv.org/abs/1701.03360
# A Dual-Stage Attention-Based recurrent neural network for time series prediction. https://arxiv.org/abs/1704.02971

import torch
import torch.nn as nn
import torch.autograd as autograd

#import torch.jit as jit

use_cuda = torch.cuda.is_available()


class LSTMCell(nn.Module): #jit.ScriptModule
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.dropout = dropout

    #@jit.script_method
    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)
        gates = (torch.matmul(input, self.weight_ih.t()) + self.bias_ih +
                 torch.matmul(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        self.dropout = dropout

    #@jit.script_method
    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        ifo_gates = (torch.matmul(input, self.weight_ii.t()) + self.bias_ii +
                     torch.matmul(hx, self.weight_ih.t()) + self.bias_ih +
                     torch.matmul(cx, self.weight_ic.t()) + self.bias_ic)
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)
        
        cellgate = torch.matmul(hx, self.weight_hh.t()) + self.bias_hh
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        if self.input_size == self.hidden_size:
          hy = outgate * (ry + input)
        else:
          hy = outgate * (ry + torch.matmul(input, self.weight_ir.t()))
        return hy, (hy, cy)


class ResLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ResLSTMCell(input_size, hidden_size, dropout=0.)

    #@jit.script_method
    def forward(self, input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        #outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []
        for i in range(len(inputs)):
            out, hidden = self.cell(inputs[i], hidden)
            outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden


class AttentiveLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
      super(AttentiveLSTMLayer, self).__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      attention_hsize = hidden_size
      self.attention_hsize = attention_hsize

      self.cell = LSTMCell(input_size, hidden_size)
      self.attn_layer = nn.Sequential(nn.Linear(2 * hidden_size + input_size, attention_hsize),
                                      nn.Tanh(),
                                      nn.Linear(attention_hsize, 1))
      self.softmax = nn.Softmax(dim=0)
      self.dropout = dropout

    #@jit.script_method
    def forward(self, input, hidden):
      inputs = input.unbind(0)
      #outputs = torch.jit.annotate(List[Tensor], [])
      outputs = []

      for t in range(len(input)):
          # attention on windows
          hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)
          hx_rep = hx.repeat(len(inputs), 1, 1)
          cx_rep = cx.repeat(len(inputs), 1, 1)
          x = torch.cat((input, hx_rep, cx_rep), dim=-1)
          l = self.attn_layer(x)
          beta = self.softmax(l)
          context = torch.bmm(beta.permute(1, 2, 0), 
                              input.permute(1, 0, 2)).squeeze(1)
          out, hidden = self.cell(context, hidden)
          outputs += [out]
      outputs = torch.stack(outputs)
      return outputs, hidden

class ODELayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout=0.0):
        
        super(ODELayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nu = input_size
        self.nx = hidden_size
        self.n_feat = 2 * hidden_size
        
        self.dt = nn.Parameter( torch.tensor(0.01) )

        # calculate network dimensions
        self.net_in = self.nu + self.nx
        self.net_out = self.nx

        # Neural network for the linear core
        self.net_dx_linear = nn.Sequential(
            nn.Linear(self.net_in, self.net_out,  bias=False))

        # Small initialization is better for multi-step methods
        for m in self.net_dx_linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)

        # Neural network for the nonlinear
        self.net_dx_nonlinear = nn.Sequential(
            nn.Linear(self.net_in, self.n_feat),
            nn.ReLU(),
            nn.Linear(self.n_feat, self.n_feat),
            nn.ReLU(),
            nn.Linear(self.n_feat, self.net_out),
        )

        # initialize with small weights and enable gradients
        for m in self.net_dx_nonlinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.constant_(m.bias, val=0)
                m.requires_grad_(True)

        self.net_output = nn.Sequential(
            nn.Linear(self.nx, self.n_feat),
            nn.ReLU(),
            nn.Linear(self.n_feat, self.n_feat),
            nn.ReLU(),
            nn.Linear(self.n_feat, self.nx),
        )

        # initialize with small weights and enable gradients
        for m in self.net_output.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.constant_(m.bias, val=0)
                m.requires_grad_(True)


    def forward(self, input, x_step):

        # number of batches and number of time steps depends on input
        nt = input.shape[0]
        
        self._dt_half = self.dt / 2.0
        self._dt_sixth = self.dt / 6.0
        
        x_step = x_step[0, :, :]
        y_all = []

        # for all time steps t
        for t in range(nt):

            # get the state and the input of one time step
            u_step = input[t, :, :]

            # compute one step of runge kutta 4
            in_xu = torch.cat((x_step, u_step), 1)
            k1 = self.net_dx_linear(in_xu) + self.net_dx_nonlinear(in_xu)
            in_xu = torch.cat((x_step + self._dt_half * k1, u_step), 1)
            k2 = self.net_dx_linear(in_xu) + self.net_dx_nonlinear(in_xu)
            in_xu = torch.cat((x_step + self._dt_half * k2, u_step), 1)
            k3 = self.net_dx_linear(in_xu) + self.net_dx_nonlinear(in_xu)
            in_xu = torch.cat((x_step + self.dt * k3, u_step), 1)
            k4 = self.net_dx_linear(in_xu) + self.net_dx_nonlinear(in_xu)
            dx = self._dt_sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            x_step = x_step + dx

            # compute output network
            y_step = self.net_output(x_step)
            y_all.append(y_step)
            
        y_all = torch.stack(y_all)
        return y_all, x_step
  
  

class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dilations, dropout=0, cell_type='GRU', batch_first=False):

        super(DRNN, self).__init__()

        self.dilations = dilations
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "ResLSTM":
            cell = ResLSTMLayer
        elif self.cell_type == "AttentiveLSTM":
            cell = AttentiveLSTMLayer
        elif self.cell_type == 'ODE':
            cell = ODELayer
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
                                                       hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM' or self.cell_type == 'ResLSTM' or self.cell_type == 'AttentiveLSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden) # compatibility hack
        
        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, autograd.Variable(zeros_)))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = autograd.Variable(torch.zeros(batch_size, hidden_dim))
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM" or self.cell_type == 'ResLSTM' or self.cell_type == 'AttentiveLSTM':
            memory = autograd.Variable(torch.zeros(batch_size, hidden_dim))
            if use_cuda:
                memory = memory.cuda()
            return hidden, memory
        else:
            return hidden


if __name__ == '__main__':
    n_inp = 4
    n_hidden = 4
    n_layers = 2
    batch_size = 3
    n_windows = 2
    cell_type = 'ODE'

    model = DRNN(n_inp, n_hidden, n_layers=n_layers, cell_type=cell_type, dilations=[1,2])

    test_x1 = torch.autograd.Variable(torch.randn(n_windows, batch_size, n_inp))

    out, hidden = model(test_x1)

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from genotypes import PRIMITIVES, Genotype
from operations import *




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MixedOp(nn.Module):

  def __init__(self, C, stride, k):
    super(MixedOp, self).__init__()
    self.k = k
    self.C = C
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)

    for primitive in PRIMITIVES:
      op = OPS[primitive](C //self.k, stride, False)
      self._ops.append(op)

  def forward(self, x, weights):
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    if self.k == 1:
      return temp1
    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)

    return ans
  
  def wider(self, k):
    self.k = k
    for op in self._ops:
      op.wider(self.C//k, self.C//k)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, k):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.k = k

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, self.k)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)
  
  def wider(self, k):
    self.k = k
    for op in self._ops:
      op.wider(k)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, k=4, grayscale=False
               ):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.k = k

    C_curr = stem_multiplier*C
    init_channels = 3
    if grayscale:
      init_channels = 1
    self.stem = nn.Sequential(
      nn.Conv2d(init_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, k)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()


  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new
  
  def show_arch_parameters(self):
    with torch.no_grad():
      logging.info('alphas normal :\n{:}'.format(F.softmax(self.alphas_normal, dim=-1).cpu()))
      logging.info('alphas reduce :\n{:}'.format(F.softmax(self.alphas_reduce, dim=-1).cpu()))




  def wider(self, k):
    self.k = k
    for cell in self.cells:
      cell.wider(k)

  def get_softmax(self):
      weights_normal = F.softmax(self.alphas_normal, dim=-1)
      weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
      return {'normal':weights_normal, 'reduce':weights_reduce}

  def get_equal_softmax(self):
    alphas_normal = nn.Parameter(1e-3 * torch.randn(self.num_edges, self.num_ops))
    alphas_reduce = nn.Parameter(1e-3 * torch.randn(self.num_edges, self.num_ops))
    weights_normal = F.softmax(alphas_normal, dim=-1)
    weights_reduce = F.softmax(alphas_reduce, dim=-1)
    return {'normal': weights_normal, 'reduce': weights_reduce}
  def get_equal_projected_weights(self, cell_type):
    ''' used in forward and genotype '''
    weights = self.get_equal_softmax()[cell_type]

    return weights

  def get_projected_weights(self, cell_type):
    ''' used in forward and genotype '''
    weights = self.get_softmax()[cell_type]


    return weights

  def forward(self, input, weights_dict=None):

    if weights_dict is None or 'normal' not in weights_dict:
      weights_normal = self.get_projected_weights('normal')
    else:
      weights_normal = weights_dict['normal']
    if weights_dict is None or 'reduce' not in weights_dict:
      weights_reduce = self.get_projected_weights('reduce')
    else:
      weights_reduce = weights_dict['reduce']

    s0 = s1 = self.stem(input)



    
    for i, cell in enumerate(self.cells):
      if cell.reduction:        
        weights = weights_reduce
      else:
        weights = weights_normal
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    loss = self._criterion(logits, target)
    return loss


  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.num_ops=num_ops
    self.num_edges=k


    self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]


  def arch_parameters(self):
    return self._arch_parameters
  
  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene


    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


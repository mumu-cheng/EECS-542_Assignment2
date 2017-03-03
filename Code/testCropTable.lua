require 'paths'
require 'nn'
paths.dofile('CropTable.lua')

mlp = nn.CropTable(1, 2)
pred = mlp:forward({torch.randn(6, 7), torch.randn(3, 7)})
print(pred)

back_pred = mlp:backward({torch.randn(6, 7), torch.randn(3, 7)}, {torch.randn(3, 7), torch.randn(3, 7)})
print(back_pred)
print(back_pred[1])
print(back_pred[2])
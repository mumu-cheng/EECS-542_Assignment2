require 'nn'
require 'nngraph'

paths.dofile('CropTable.lua')

fcn_net = nn.Sequential()

local layer_stack1 = nn.Sequential()
local layer_stack2 = nn.Identity()

local function ConvReLU1(nInputPlane, nOutputPlane)
  layer_stack_1:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  layer_stack_1:add(nn.ReLU(true))
  return layer_stack_1
end

layer_stack_1:add(nn.SpatialConvolution(3,64,3,3,1,1,100,100))--conv1_1
layer_stack_1:add(nn.ReLU(true))-- relu1_1
ConvReLU1(64,64) -- conv1_1 relu1_2
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))-- pool1
ConvReLU1(64,128) -- conv2_1 relu2_1
ConvReLU1(128,128) -- conv2_2 relu2_2
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2)-- pool2
ConvReLU1(128,256) -- conv3_1 relu3_1
ConvReLU1(256,256) -- conv3_2 relu3_2
ConvReLU1(256,256) -- conv3_3 relu3_3 
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))--pool3
ConvReLU1(256,512) -- conv4_1 relu4_1
ConvReLU1(512,512) -- conv4-2 relu4_2
ConvReLU1(512,512) -- conv4_3 relu4_3
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))--pool4
ConvReLU1(512,512) -- conv5_1 relu5_1
ConvReLU1(512,512) -- conv5_2 relu5_2
ConvReLU1(512,512) -- conv5_3 relu5_3
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))--pool5
layer_stack_1:add(nn.SpatialConvolution(512,4096,7,7,1,1,0,0))--fc6
layer_stack1:add(nn.ReLU(true))--relu6
layer_stack1:add(nn.Dropout(0.5)) --drop6
layer_stack1:add(nn.SpatialConvolution(4096,4096,1,1,1,1,0,0))--fc7
layer_stack1:add(nn.ReLU(true)) -- relu7
layer_stack1:add(nn.Dropout(0.5)) --drop7
layer_stack1:add(nn.SpatialConvolution(4096,21,1,1,1,1,0,0)) --score_fr stride?
layer_stack1:add(nn.SpacialFullConvolution(21,21,64,64,32,32,0,0,0,0):noBias())) -- upscore

fcn_net:add(nn.ConcatTable()
            :add(layer_stack_1)
            :add(layer_stack_2))

fcn_net:add(nn.CropTable(2, 20))
fcn_net:add(nn.CropTable(3, 20))
fcn_net:add(nn.SelectTable(1))

-- convert the net to cudnn
cudnn.convert(fcn_net, cudnn)
--loss
crit = cudnn.SpatialCrossEntropyCriterion()


-- initialization from MSR
-- local function MSRinit(net)
--   local function init(name)
--     for k,v in pairs(net:findModules(name)) do
--       local n = v.kW*v.kH*v.nOutputPlane
--       v.weight:normal(0,math.sqrt(2/n))
--       v.bias:zero()
--     end
--   end
--   -- have to do for both backends
--   init'nn.SpatialConvolution'
-- end

-- MSRinit(vgg)

trainer = nn.StochasticGradient(fcn_net, crit)
trainer.learningRate = 0.001
trainer.maxIteration = 5
-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return fcn_net

require 'nn'
require 'nngraph'
-- require 'cunn'
-- require 'cudnn'

-- primary net
fcn_net = nn.Sequential()
-- sub_branch_1 from pool3 to upscore_pool4
local layer_stack_1 = nn.Sequential()
-- sub_branch_1 from pool4 to upscore2
local layer_stack_2 = nn.Sequential()
local layer_stack_3 = nn.Sequential()

-- building block for fcn_net
local function ConvReLU(nInputPlane, nOutputPlane)
  fcn_net:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  fcn_net:add(nn.ReLU(true))
  return fcn_net
end
--upscore2
layer_stack_1:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())
--score_pool4
layer_stack_1:add(nn.SpatialConvolution(512,21,1,1,1,1,0,0))
--score_pool4c(crop score_pool4 to upscore2)
layer_stack_1:add(nn.Narrow(2,6,-6))
layer_stack_1:add(nn.Narrow(3,6,-6))

layer_stack_2:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())

-- conv1_1 & relu1_1
fcn_net:add(nn.SpatialConvolution(3,64,3,3,1,1,100,100))
fcn_net:add(nn.ReLU(true))
-- conv1_2 & relu1_1
ConvReLU(64,64)
-- pool1
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv2_1 & relu2_1
ConvReLU(64,128)
-- conv2_2 & relu2_2
ConvReLU(128,128)
-- pool2
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv3_1 && relu3_1
ConvReLU(128,256)
-- conv3_2 && relu3_2
ConvReLU(256,256)
-- conv3_3 && relu3_3
ConvReLU(256,256)
-- pool3 
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv4_1 & relu4_1
ConvReLU(256,512)
-- conv4_2 & relu4_2
ConvReLU(512,512)
-- conv4_3 & relu4_3
ConvReLU(512,512)
-- pool4
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))

-- conv5_1 & relu5_1
ConvReLU(512,512)
-- conv5_2 & relu5_2
ConvReLU(512,512)
-- conv5_3 & relu5_3
ConvReLU(512,512)
-- pool5
fcn_net:add(nn.SpatialMaxPooling(2,2,2,2))
-- fc6
fcn_net:add(nn.SpatialConvolution(512,4096,7,7,1,1,0,0))
-- relu6
fcn_net:add(nn.ReLU(true))
-- drop6
fcn_net:add(nn.Dropout(0.5))
-- fc7
fcn_net:add(nn.SpatialConvolution(4096,4096,1,1,1,1,0,0))
-- drop7
fcn_net:add(nn.Dropout(0.5))
-- score_fr
fcn_net:add(nn.SpatialConvolution(4096,21,1,1,1,1,0,0))
-- upscore2
fcn_net:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())


-- fuse_pool4
fcn:add(nn.ConcatTable()
            :add(layer_stack_1)
            :add(layer_stack_2)
fcn_n:add(nn.CAddTable(true))
--upscore16
fcn_net:add(nn.SpatialFullConvolution(21,21,16,16,8,8,0,0,0,0):noBias())
--score(crop upscore8 to data)
fcn_net:add(nn.Narrow(2,28,-28))
fcn_net:add(nn.Narrow(3,28,-28))
--loss
crit = nn.CrossEntropyCriterion()

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

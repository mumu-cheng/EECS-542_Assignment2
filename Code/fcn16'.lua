require 'nn'
require 'nngraph'
require 'path'
require 'cunn'

paths.dofile('CropTable.lua')

-- primary net
fcn_net = nn.Sequential()

local layer_stack_1 = nn.Sequential()
local layer_stack_2 = nn.Sequential()
local layer_stack_3 = nn.Sequential()
local layer_stack_4 = nn.Identity()

-- building block for fcn_net
local function ConvReLU1(nInputPlane, nOutputPlane)
  fcn_net:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  fcn_net:add(nn.ReLU(true))
  return fcn_net
end


-- conv1_1 & relu1_1
layer_stack_1:add(nn.SpatialConvolution(3,64,3,3,1,1,100,100))
layer_stack_1:add(nn.ReLU(true))
-- conv1_2 & relu1_1
ConvReLU1(64,64)
-- pool1
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv2_1 & relu2_1
ConvReLU1(64,128)
-- conv2_2 & relu2_2
ConvReLU1(128,128)
-- pool2
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv3_1 && relu3_1
ConvReLU1(128,256)
-- conv3_2 && relu3_2
ConvReLU1(256,256)
-- conv3_3 && relu3_3
ConvReLU1(256,256)
-- pool3 
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv4_1 & relu4_1
ConvReLU1(256,512)
-- conv4_2 & relu4_2
ConvReLU1(512,512)
-- conv4_3 & relu4_3
ConvReLU1(512,512)
-- pool4
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))

-- conv5_1 & relu5_1
ConvReLU2(512,512)
-- conv5_2 & relu5_2
ConvReLU2(512,512)
-- conv5_3 & relu5_3
ConvReLU2(512,512)
-- pool5
layer_stack_2:add(nn.SpatialMaxPooling(2,2,2,2))
-- fc6
layer_stack_2:add(nn.SpatialConvolution(512,4096,7,7,1,1,0,0))
-- relu6
layer_stack_2:add(nn.ReLU(true))
-- drop6
layer_stack_2:add(nn.Dropout(0.5))
-- fc7
layer_stack_2:add(nn.SpatialConvolution(4096,4096,1,1,1,1,0,0))
-- drop7
layer_stack_2:add(nn.Dropout(0.5))
-- score_fr
layer_stack_2:add(nn.SpatialConvolution(4096,21,1,1,1,1,0,0))
-- upscore2
layer_stack_2:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())

--score_pool4
layer_stack_3:add(nn.SpatialConvolution(512,21,1,1,1,1,0,0))

--fuse_pool4(fuse_pool4)
layer_stack_1:add(nn.ConcatTable()
                :add(layer_stack_3)
                :add(layer_stack_2))

--score_pool4c(crop score_pool4 to upscore2)
layer_stack_1:add(nn.CropTable(2, 6))
layer_stack_1:add(nn.CropTable(3, 6))

layer_stack_1:add(nn.CAddTable(true))

--upscore16
layer_stack_1:add(nn.SpatialFullConvolution(21,21,16,16,8,8,0,0,0,0):noBias())

--score(crop upscore8 to data)
fcn_net:add(nn.ConcatTable()
            :add(layer_stack_1)
            :add(layer_stack_4))

fcn_net:add(nn.CropTable(2, 28))
fcn_net:add(nn.CropTable(3, 28))
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

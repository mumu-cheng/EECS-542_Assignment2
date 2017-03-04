require 'nn'
require 'nngraph'

paths.dofile('CropTable.lua')
fcn_net = nn.Sequential()

local layer_stack_0 = nn.Sequential()

local layer_stack_1 = nn.Identity()

-- sub_branch_1 from pool3 to upscore_pool4
local function ConvReLU(nInputPlane, nOutputPlane)
  layer_stack_0:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  layer_stack_0:add(nn.ReLU(true))
  return layer_stack_0
end

layer_stack_0:add(nn.SpatialConvolution(3,64,3,3,1,1,100,100))--conv1_1
layer_stack_0:add(nn.ReLU(true))-- relu1_1
ConvReLU(64,64) -- conv1_1 relu1_2
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))-- pool1
ConvReLU(64,128) -- conv2_1 relu2_1
ConvReLU(128,128) -- conv2_2 relu2_2
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))-- pool2
ConvReLU(128,256) -- conv3_1 relu3_1
ConvReLU(256,256) -- conv3_2 relu3_2
ConvReLU(256,256) -- conv3_3 relu3_3 
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))--pool3
ConvReLU(256,512) -- conv4_1 relu4_1
ConvReLU(512,512) -- conv4_2 relu4_2
ConvReLU(512,512) -- conv4_3 relu4_3
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))--pool4
ConvReLU(512,512) -- conv5_1 relu5_1
ConvReLU(512,512) -- conv5_2 relu5_2
ConvReLU(512,512) -- conv5_3 relu5_3
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))--pool5
layer_stack_0:add(nn.SpatialConvolution(512,4096,7,7,1,1,0,0))--fc6
layer_stack_0:add(nn.ReLU(true))--relu6
layer_stack_0:add(nn.Dropout(0.5)) --drop6
layer_stack_0:add(nn.SpatialConvolution(4096,4096,1,1,1,1,0,0))--fc7
layer_stack_0:add(nn.ReLU(true)) -- relu7
layer_stack_0:add(nn.Dropout(0.5)) --drop7
layer_stack_0:add(nn.SpatialConvolution(4096,21,1,1,1,1,0,0)) --score_fr stride?
layer_stack_0:add(nn.SpatialFullConvolution(21,21,64,64,32,32,0,0,0,0):noBias()) -- upscore

fcn_net:add(nn.ConcatTable()
            :add(layer_stack_0)
            :add(layer_stack_1))

fcn_net:add(nn.CropTable(2, 20))
fcn_net:add(nn.CropTable(3, 20))
fcn_net:add(nn.SelectTable(1))

crit = nn.CrossEntropyCriterion()---softmax loss function
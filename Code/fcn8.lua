require 'nn'
require 'paths'
require 'cunn'
require 'cudnn'

paths.dofile('CropTable.lua')
-- primary net
fcn_net = nn.Sequential()

local layer_stack_0 = nn.Sequential()
-- sub_branch_1 from pool3 to upscore_pool4
local layer_stack_1 = nn.Sequential()
-- sub_branch_1 from pool4 to upscore2
local layer_stack_2 = nn.Sequential()
-- sub_branch_2 from pool4 to upscore2
local layer_stack_3 = nn.Sequential()
-- sub_branch_2 from pool3 to upscore_pool4
local layer_stack_4 = nn.Sequential()


local layer_stack_5 = nn.Identity()

-- building block for layer_stack_0
local function ConvReLU(nInputPlane, nOutputPlane)
  layer_stack_0:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  layer_stack_0:add(nn.ReLU(true))
  return layer_stack_0
end

-- building block for layer_stack_1
local function ConvReLU_1(nInputPlane, nOutputPlane)
  layer_stack_1:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  layer_stack_1:add(nn.ReLU(true))
  return layer_stack_1
end

-- building block for layer_stack_2
local function ConvReLU_2(nInputPlane, nOutputPlane)
  layer_stack_2:add(nn.SpatialConvolution(nInputPlane,nOutputPlane,3,3,1,1,1,1))
  -- 3x3 convolution kernel, stride of 1, pad of 1
  layer_stack_2:add(nn.ReLU(true))
  return layer_stack_2
end

-- conv4_1 & relu4_1
ConvReLU_1(256,512)
-- conv4_2 & relu4_2
ConvReLU_1(512,512)
-- conv4_3 & relu4_3
ConvReLU_1(512,512)
-- pool4
layer_stack_1:add(nn.SpatialMaxPooling(2,2,2,2))

-- conv5_1 & relu5_1
ConvReLU_2(512,512)
-- conv5_2 & relu5_2
ConvReLU_2(512,512)
-- conv5_3 & relu5_3
ConvReLU_2(512,512)
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
-- relu7
layer_stack_2:add(nn.ReLU(true))
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
layer_stack_1:add(nn.CropTable({2, 3}, {6, 6}))

layer_stack_1:add(nn.CAddTable())
--upscore_l4(upscore_pool4)
layer_stack_1:add(nn.SpatialFullConvolution(21,21,4,4,2,2,0,0,0,0):noBias())

--score_pool3
layer_stack_4:add(nn.SpatialConvolution(256,21,1,1,1,1,0,0))

-- conv1_1 & relu1_1
layer_stack_0:add(nn.SpatialConvolution(3,64,3,3,1,1,100,100))
layer_stack_0:add(nn.ReLU(true))
-- conv1_2 & relu1_1
ConvReLU(64,64)
-- pool1
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv2_1 & relu2_1
ConvReLU(64,128)
-- conv2_2 & relu2_2
ConvReLU(128,128)
-- pool2
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))
-- conv3_1 && relu3_1
ConvReLU(128,256)
-- conv3_2 && relu3_2
ConvReLU(256,256)
-- conv3_3 && relu3_3
ConvReLU(256,256)
-- pool3 
layer_stack_0:add(nn.SpatialMaxPooling(2,2,2,2))
-- fuse_pool3
layer_stack_0:add(nn.ConcatTable()
            :add(layer_stack_4)
            :add(layer_stack_1))

--score_pool3c(crop score_pool3 to upscore_pool4)
layer_stack_0:add(nn.CropTable({2, 3}, {10, 10}))
layer_stack_0:add(nn.CAddTable(true))

--upscore8
layer_stack_0:add(nn.SpatialFullConvolution(21,21,16,16,8,8,0,0,0,0):noBias())
--score(crop upscore8 to data)
fcn_net:add(nn.ConcatTable()
            :add(layer_stack_0)
            :add(layer_stack_5))

fcn_net:add(nn.CropTable({2, 3}, {32, 32}))
fcn_net:add(nn.SelectTable(1))
fcn_net:add(nn.Unsqueeze(1))
fcn_net:add(nn.Contiguous())
-- convert the net to cudnn
-- cudnn.convert(fcn_net, cudnn)

return fcn_net
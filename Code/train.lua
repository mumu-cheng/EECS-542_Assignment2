require 'nn'
require 'nngraph'
require 'paths'
require 'cunn'
require 'cudnn'
require 'optim'

-- write the loss to a text file and read from there to plot the loss as training proceeds
logger = optim.Logger('loss_log.txt')

-- states variables for the optimization process
local optimState = {
	learningRate = 0.0001, -- learning rate 10^-4 as per the paper
	-- learningRateDecay = 1e-4,
	momentum = 0.99,
	weightDecay = 5^-4
}
-- hyperparameter
local config = {
	batch_size = 20,
	max_epoch = 736 -- max number of epochs
}
-- train 
function train()
	-- load net module
	paths.dofile('FCN8.lua')
	fcn_net = fcn_net:cuda()
	print(fcn_net)
	-- criterion for loss
	criterion = cudnn.SpatialCrossEntropyCriterion() -- how to implement 'normalize: false'
	criterion = criterion:cuda()
	-- trainset 
	trainset = 
	trainset.data = trainset.data:cuda()
	trainset.label = trainset.label:cuda()

	-- start to train the net
	params, gradParams = fcn_net:getParameters()

	for epoch = 1, config.max_epoch do
   	-- local function we give to optim
   	-- it takes current weights as input, and outputs the loss
   	-- and the gradient of the loss with respect to the weights
   	-- gradParams is calculated implicitly by calling 'backward',
   	-- because the model's weight and bias gradient tensors
   	-- are simply views onto gradParams
   		cur_loss = 0
   		for 
	   		function feval(params)
	      		gradParams:zero()
				local outputs = fcn_net:forward(batchInputs)
				-- ignore_label: 255; pixels of 255 are not counted into loss function

	      		local loss = criterion:forward(outputs, batchLabels)
	      		local dloss_doutputs = criterion:backward(outputs, batchLabels)
	      		fcn_net:backward(batchInputs, dloss_doutputs)
	      		return loss, gradParams
	   		end
	   		_, batch_loss = optim.sgd(feval, params, optimState)
	   		cur_loss = cur_loss + batch_loss[1]
	   	end
   		print('-----------------------------------------------------------')
   		print('epoch = ' .. epoch .. ',    current loss = ' .. current_loss)
   		-- write the loss since this epoch to the log
   		logger:add{['training error'] = current_loss}
   		logger:style{['training error'] = '-'}
   		logger:plot() 
	end

	-- example for dataset
	-- dataset={};
	-- function dataset:size() return 100 end -- 100 examples
	-- for i=1,dataset:size() do 
	--   local input = torch.randn(2);     -- normally distributed example in 2d
	--   local output = torch.Tensor(1);
	--   if input[1]*input[2]>0 then     -- calculate label for XOR function
	--     output[1] = -1;
	--   else
	--     output[1] = 1
	--   end
	--   dataset[i] = {input, output}
	-- end
end

-- functions to calculate all four metrics
-- pixel accuracy
local function cal_pixel_accuracy()

end
-- pixel accuracy 
local function cal_mean_accuracy()
end
-- mean IU
local function cal_iu()
end
-- frequency weighted IU
local function cal_fw_iu()
end

-- test
function test()
	testset = 

	for 1=1,10000 do
		local true_seg = 
		local net_seg = fcn_net:forward(testset.data[100])
	end
	for i = 1,(#data)[1] do
	   local myPrediction = model:forward(data[i][{{2,3}}])
	   print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
	end
end

train()
test()
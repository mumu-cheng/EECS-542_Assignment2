require 'nn'
require 'nngraph'
require 'paths'
require 'cunn'
require 'cudnn'
require 'optim'
require 'cutorch'

-- load the data
-- paths.dofile('load.lua')
print('>>>> Start loading training dataset')
trainset = torch.load('../Datasett7/trainset.t7')
function trainset:size()
    return #self
end

function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end


function trainset:shuffle()
	local counter = trainset:size()
	while counter > 1 do
		local index = math.random(counter)
		swap(trainset, index, counter)
		counter = counter - 1
	end
end

function trainset:resize(ratio)
	for i = 1, trainset:size() do
		local H = trainset[i]:size(2)
		local W = trainset[i]:size(3)
		local new_H = torch.floor(ratio * H)
		local new_W = torch.floor(ratio * W)
		trainset[i][1] = image.scale(trainset[i][1], new_H, new_W, "simple")
		trainset[i][2] = image.scale(trainset[i][2], new_H, new_W, "simple")
	end
end

trainset:resize(0.5)

print('>>>> Start loading validation dataset')
-- valset = torch.load('../Datasett7/valset.t7')
print('>>>> Finish loading dataset')
-- convert the data to cuda (more than 12.7 GB out of memery)
-- paths.dofile('convertDataToCuda.lua')
-- convertdatatocuda()
-- load the functions to calculate metrics 
paths.dofile('metrics.lua')
-- write the loss to a text file and read from there to plot the loss as training proceeds
logger = optim.Logger('train_loss.log')
logger:setNames{'Epoch','Training loss.','Val pixel acc.','Val mean acc.','mean IU','FW IU'}
-- states variables for the optimization process
local optimState = {
	learningRate = 0.0001, -- learning rate 10^-4 as per the paper
	-- learningRateDecay = 1e-4,
	momentum = 0.90,
	weightDecay = 0.0005
}
-- hyperparameter
local config = {
	batch_size = 1, -- online learning
	max_epoch = 2, -- max number of epochs
	trainset_size = trainset:size(),
	-- valset_size = valset:size(),
	-- testset_size = testset:size()
}

-- load net module
paths.dofile('fcn8.lua')
fcn_net = fcn_net:cuda()
print('>>>> Finish loading net and converting net to cuda')
local free, total = cutorch.getMemoryUsage()
print(free, total)
-- print(fcn_net)

-- train 
function train()
	-- criterion for loss
	criterion = cudnn.SpatialCrossEntropyCriterion() -- how to implement 'normalize: false'
	criterion = criterion:cuda()
	-- start to train the net
	local params, gradParams = fcn_net:getParameters()
	for epoch = 1, config.max_epoch do
		print('>>>> Starting to train epoch ' .. epoch .. ':')
   		cur_loss = 0
   		for iter = 1, config.trainset_size do
	   		function feval(params)
	   			-- fcn_net:zeroGradParameters()
	      		gradParams:zero()
				batchInputs = trainset[iter][1]:cuda()
				batchLabels = trainset[iter][2]:cuda()
				print(batchInputs:size())
				-- batchInputs = batchInputs:cuda()
				-- batchLabels = batchLabels:cuda()
				local outputs = fcn_net:forward(batchInputs)
				-- ignore_label: 255; pixels of 255 are not counted into loss function
				if torch.max(batchLabels) == 255 then
					local idx = batchLabels:eq(255)
					batchLabels[idx] = 1
					-- idx = idx:squeeze()
					outputs[1][1][idx] = 1
					for j = 2,21 do
						outputs[1][j][idx] = 0
					end
				end
				-- calculate loss
	      			local loss = criterion:forward(outputs, batchLabels)
	      			local dloss_doutputs = criterion:backward(outputs, batchLabels)
	      			fcn_net:backward(batchInputs, dloss_doutputs)
	      			return loss, gradParams
	   		end
			collectgarbage()
			collectgarbage()
	   		local free, total = cutorch.getMemoryUsage()
			print(free, total)
	   		_, loss = optim.sgd(feval, params, optimState)
	   		print('>>>> iter = '.. iter.. ', per-image loss = ')--.. loss[1])
	   		-- save the preliminary model
			-- torch.save('fcn8.t7', fcn_net)
	   		cur_loss = cur_loss + loss[1]
	   	end
   		print('>>>> Epoch = '.. epoch.. ', current loss = '.. cur_loss)
   		-- val(epoch)
   		-- write the loss since this epoch to the log
   		logger:add{epoch, current_loss, acc, mean_acc, mean_iu, fw_iu}
   		-- logger:style{'+-','+-','+-','+-','+-','+-'}   		
		trainset:shuffle()
	end
	-- logger:plot() 
end

-- validate
function val(epoch)
	print('>>>> Validation for epoch '.. epoch)
	softmax_layer = nn.SpatialSoftMax():cuda()
	for i = 1, config.valset_size do 
		val_image = valset[i][1]:cuda()
		true_seg = valset[i][2]:cuda()
		net_seg = fcn_net:forward(val_image)
		net_seg = softmax_layer:forward(net_seg)
		_, net_seg = torch.max(net_seg,2)
		net_seg = net_seg:squeeze()
		true_seg = true_seg:squeeze()
		compute_hist(net_seg,true_seg) 
	end
	calculate_metrics()
end

-- test(need to modify)
function test()
	softmax_layer = nn.SpatialSoftMax():cuda()
	for i = 1, config.testset_size do 
		test_image = testset[i][1]:cuda()
		true_seg = testset[i][2]:cuda()
		net_seg = fcn_net:forward(test_image)
		net_seg = softmax_layer:forward(net_seg)
		compute_hist(net_seg,true_seg)
	end
	epoch = 'test phase'
	calculate_metrics()
end

-- run
train()
-- test()

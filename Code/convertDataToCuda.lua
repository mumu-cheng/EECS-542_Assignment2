require "nn"
require "cunn"
require "cudnn"

for i in 1, trainset:size() do
    trainset[i][1] = trainset[i][1]:cuda()
    trainset[i][2] = trainset[i][2]:cuda()
end

for i in 1, valset:size() do
    valset[i][1] = valset[i][1]:cuda()
    valset[i][2] = valset[i][2]:cuda()
end

for i in 1, testset:size() do
    testset[i] = testset[i]:cuda()
end
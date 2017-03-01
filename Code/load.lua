-- Load image from the VOC2011 image sets

-- Load the train set
require 'image'
train_indices = {}
val_indices = {}
test_indices = {}

trainset = {}
valset = {}
testset = {}
local train_f = io.open("./VOC2011/ImageSets/Segmentation/train.txt")
if train_f then
    for line in train_f:lines() do
        table.insert(train_indices, line)
    end
else
end


for i = 1, #train_indices do
    sample = {}
    train_img_file = './VOC2011/JPEGImages/'..train_indices[i]..'.jpg';
    label_file = './VOC2011/SegmentationClass/'..train_indices[i]..'.png'
    local train_img = image.load(train_img_file, 3, 'byte')
    local label = image.load(label_file, 1, 'byte')
    table.insert(sample, train_img)
    table.insert(sample, label)
    table.insert(trainset, sample)
end

function trainset:size()
    return #self
end

-- load the validate set
local val_f = io.open("./VOC2011/ImageSets/Segmentation/val.txt")
if val_f then
    for line in val_f:lines() do
        table.insert(val_indices, line)
    end
else
end

for i = 1, #val_indices do
    val_img_file = './VOC2011/JPEGImages/'..val_indices[i]..'.jpg';
    local val_img = image.load(val_img_file, 3, 'byte')
    table.insert(valset, val_img)
end

-- load the test set
local test_f = io.open("./VOC2011/ImageSets/Segmentation/test.txt")
if test_f then
    for line in test_f:lines() do
        table.insert(test_indices, line)
    end
else
end

for i = 1, #test_indices do
    test_img_file = './VOC2011/JPEGImages/'..test_indices[i]..'.jpg';
    local test_img = image.load(test_img_file, 3, 'byte')
    table.insert(testset, test_img)
end

-- TODO
-- preprocess the image
-- cast to float
-- substract the mean
-- switch channels RGB -> BGR






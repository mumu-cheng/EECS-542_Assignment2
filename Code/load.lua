require 'paths'
-- preprocess the image
-- cast to double
-- switch channels RGB -> BGR
-- substract the mean
function preprocess(img)
    mean = {104.00698793, 116.66876762, 122.67891434}
    new_img = (img:clone()):double()
    new_img[{{1}, {}, {}}] = img[{{3}, {}, {}}]
    new_img[{{3}, {}, {}}] = img[{{1}, {}, {}}]
    for i = 1,3 do
        new_img[{{i}, {}, {}}]:add(-mean[i])
    end
    return new_img
end

paths.dofile('getLabelMap.lua')
function convert_label(label)
    local height = label:size()[2]
    local width = label:size()[3]
    new_label = torch.zeros(height, width)
    for i = 1, height do
        for j = 1, width do
            local channel1 = label[1][i][j]
            local channel2 = label[2][i][j]
            local channel3 = label[3][i][j]
            new_label[i][j] = labelMap[channel1 * 255 * 255 + channel2 * 255 + channel3]
        end
    end
    return new_label
end

data_dir = "../"

-- Load image from the VOC2011 image sets

-- Load the train set
require 'image'

print("Start loading data...")
train_indices = {}
val_indices = {}
test_indices = {}

trainset = {}
valset = {}
testset = {}
local train_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/train.txt")
if train_f then
    for line in train_f:lines() do
        table.insert(train_indices, line)
    end
else
end

print("Loading train data...")
for i = 1, #train_indices do
    if (i % 10 == 0) then
        print("Progress: "..tostring(math.floor(i * 100/#train_indices)).."%")
    end
    sample = {}
    train_img_file = data_dir..'VOC2011/JPEGImages/'..train_indices[i]..'.jpg';
    label_file = data_dir..'VOC2011/SegmentationClass/'..train_indices[i]..'.png'
    local train_img = preprocess(image.load(train_img_file, 3, 'byte'))
    local label = convert_label(image.load(label_file, 3, 'byte'))
    table.insert(sample, train_img)
    table.insert(sample, label)
    table.insert(trainset, sample)
end
print("Finish loading")

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


-- load the validate set
local val_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/val.txt")
if val_f then
    for line in val_f:lines() do
        table.insert(val_indices, line)
    end
else
end

print("Loading validation data...")
for i = 1, #val_indices do
    if (i % 10 == 0) then
        print("Progress: "..tostring(math.floor(i * 100/#val_indices)).."%")
    end
    sample = {}
    val_img_file = data_dir..'VOC2011/JPEGImages/'..val_indices[i]..'.jpg';
    val_label_file = data_dir..'VOC2011/SegmentationClass/'..val_indices[i]..'.png'
    local val_img = preprocess(image.load(val_img_file, 3, 'byte'))
    local label = convert_label(image.load(val_label_file, 3, 'byte'))
    table.insert(sample, val_img)
    table.insert(sample, label)
    table.insert(valset, sample)
end
print("Finish loading")

function valset:size()
    return #self
end

-- load the test set
local test_f = io.open(data_dir.."VOC2011/ImageSets/Segmentation/test.txt")
if test_f then
    for line in test_f:lines() do
        table.insert(test_indices, line)
    end
else
end

print("Loading test data...")
for i = 1, #test_indices do
    if (i % 10 == 0) then
        print("Progress: "..tostring(math.floor(i * 100/#test_indices)).."%")
    end
    test_img_file = data_dir..'VOC2011/JPEGImages/'..test_indices[i]..'.jpg';
    local test_img = preprocess(image.load(test_img_file, 3, 'byte'))
    table.insert(testset, test_img)
end
print("Finish loading")


function testset:size()
    return #self
end

print("Finish loading data")


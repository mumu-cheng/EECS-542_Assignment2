local CropTable, Parent = torch.class('nn.CropTable', 'nn.Module')

function CropTable:__init(axis, offset)
    Parent.__init(self)
    self.axis = axis
    self.offset = offset
end

-- input should be like {tensor1, tensor2}, crop tensor1 to be the same size as tensor2
-- on the axis with certain offset
function CropTable:updateOutput(input)
    self.output = {input[1]:narrow(self.axis, self.offset, input[2]:size()[self.axis]), input[2]}
    return self.output
end

-- gradInput should be like {tensor1, tensor2}, where gradInput[1] should be the same size as input[1](scale back)
function CropTable:updateGradInput(input, gradOutput)
    leftPaddingSize = input[2]:size()
    leftPaddingSize[self.axis] = self.offset - 1


    rightPaddingSize = input[2]:size()
    rightPaddingSize[self.axis] = input[1]:size()[self.axis] - input[2]:size()[self.axis] - self.offset + 1

    leftPadding = torch.zeros(leftPaddingSize)
    rightPadding = torch.zeros(rightPaddingSize)

    self.gradInput = {torch.cat({leftPadding, gradOutput[1], rightPadding}, self.axis), gradOutput[2]}
    return self.gradInput
end
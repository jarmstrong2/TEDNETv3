require 'nn'

local Inverse, parent = torch.class('nn.Inverse', 'nn.Module')

function Inverse:updateOutput(input)
    batchSize = input:size()[1]
    dim = input:size()[2]
    self.output = torch.CudaTensor(batchSize, dim, dim)
    for i = 1, batchSize do
        inputSize = ((input[i]):size())[1]
        --eps = torch.eye(inputSize):float() * 1e-1
        self.output[i] = (torch.inverse(input:float()[i])):cuda()
    end
    return self.output
end

function Inverse:updateGradInput(input, gradOutput)
    batchSize = input:size()[1]
    dim = input:size()[2]
    self.gradInput =  torch.CudaTensor(batchSize, dim, dim)
    for i = 1, batchSize do
        inputSize = ((input[i]):size())[1]
        --eps = torch.eye(inputSize):float() * 1e-1  
        self.gradInput[i] = -((torch.inverse(input:float()[i])):cuda() * 
            gradOutput[i]) * (torch.inverse(input:float()[i])):cuda()
    end
    return self.gradInput
end
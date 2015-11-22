require 'nn'

local LogDeterminant, parent = torch.class('nn.LogDeterminant', 'nn.Module')

function LogDeterminant:updateOutput(input)
    local batchSize = input:size()[1]
    self.output = torch.CudaTensor(batchSize, 1)
    for i = 1, batchSize do
        local inputSize = ((input[i]):size())[1]
        --eps = torch.eye(inputSize):float() * 1e-2
        local eig_vals = torch.eig(input[i], 'N')
        self.output[i] = (torch.log(eig_vals:select(2, 1))):sum()
    end
    return self.output
end

function LogDeterminant:updateGradInput(input, gradOutput)
    local batchSize = input:size()[1]
    self.gradInput =  torch.CudaTensor(input:size())
    for i = 1, batchSize do  
        local inputSize = ((input[i]):size())[1]
        --eps = torch.eye(inputSize):float() * 1e-2
        local invInput = torch.inverse(input[i]) 
        self.gradInput[i] = invInput:t() * gradOutput[i]:squeeze()
    end
    return self.gradInput
end

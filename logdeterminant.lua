require 'nn'

local LogDeterminant, parent = torch.class('nn.LogDeterminant', 'nn.Module')

function LogDeterminant:updateOutput(input)
    batchSize = input:size()[1]
    self.output = torch.zeros(batchSize, 1)
    for i = 1, batchSize do
        inputSize = ((input[i]):size())[1]
        eps = torch.eye(inputSize) * 1e-2
        eig_vals = torch.eig(input[i] + eps, 'N')
        self.output[i] = torch.log(eig_vals:select(2, 1):sum())
    end
    return self.output
end

function LogDeterminant:updateGradInput(input, gradOutput)
    batchSize = input:size()[1]
    self.gradInput =  torch.zeros(input:size())
    for i = 1, batchSize do  
        inputSize = ((input[i]):size())[1]
        eps = torch.eye(inputSize) * 1e-2
        invInput = torch.inverse(input[i] + eps)
        self.gradInput[i] = invInput:t() * gradOutput[i]:squeeze()
    end
    return self.gradInput
end
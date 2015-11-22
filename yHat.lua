require 'nn'

local YHat, parent = torch.class('nn.YHat', 'nn.Module')

function YHat:__init()
   parent.__init(self)
   self.sizeMeanInput = opt.inputSize * opt.numMixture

   -- if flag opt.isCovarianceFull true then input represents fill covariance
   if opt.isCovarianceFull then
        self.sizeCovarianceInput = opt.inputSize * opt.numMixture * opt.dimSize
   
   -- otherwise the input represents the main axis of a diagonal covariance
   else
        self.sizeCovarianceInput = opt.inputSize * opt.numMixture
   end
end

function YHat:updateOutput(input)
    local piStart = 1
    local piEnd = opt.numMixture
    local hat_pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local hat_mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local hat_sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    if opt.isCovarianceFull then
        sigma_t = hat_sigma_t
    else
        self.sigma_t_act = self.sigma_t_act or nn.Exp():cuda()
        sigma_t = self.sigma_t_act:forward(hat_sigma_t)
    end

    self.pi_t_act = self.pi_t_act or nn.LogSoftMax():cuda()
   
    local pi_t = self.pi_t_act:forward(hat_pi_t)
    local mu_t = hat_mu_t:clone()

    self.output = {pi_t, mu_t, sigma_t}
    
    return self.output
end

function YHat:updateGradInput(input, gradOutput)
    local piStart = 1
    local piEnd = opt.numMixture
    local hat_pi_t = input[{{},{piStart,piEnd}}]

    local muStart = piEnd + 1
    local muEnd = piEnd + self.sizeMeanInput
    local hat_mu_t = input[{{},{muStart,muEnd}}]

    local sigmaStart = muEnd + 1
    local sigmaEnd = muEnd + self.sizeCovarianceInput
    local hat_sigma_t = input[{{},{sigmaStart,sigmaEnd}}]

    local d_hat_pi_t, d_hat_mu_t, d_hat_sigma_t, _ = 
    unpack(gradOutput)

    local grad_hat_pi_t = self.pi_t_act:backward(hat_pi_t, d_hat_pi_t:clone())
    local grad_hat_mu_t = d_hat_mu_t:clone()
   
   if opt.isCovarianceFull then
        grad_hat_sigma_t = d_hat_sigma_t
    else
        grad_hat_sigma_t = self.sigma_t_act:backward(hat_sigma_t,d_hat_sigma_t)
    end
    
    local grad_input = torch.cat(grad_hat_pi_t:float(), grad_hat_mu_t:float(), 2)
    grad_input = torch.cat(grad_input, grad_hat_sigma_t:float(), 2)
    
    self.gradInput = grad_input:cuda() 

    return self.gradInput  
end

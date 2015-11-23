require 'cutorch'
require 'cunn'
require 'nn'
require 'nngraph'
require 'torch'
require 'logdeterminant'
require 'inverse'

local mixture = {}

function mixture.gauss(inputSize, uDimSize, nMixture)
    local target = nn.Identity()()
    local pi = nn.Identity()()
    local mu = nn.Identity()()
    local u = nn.Identity()()
    local mask = nn.Identity()()
    local eps = nn.Identity()()

    local u_reshaped = nn.Reshape(nMixture, uDimSize, inputSize)(u)
    local u_pack = nn.SplitTable(2,4)(u_reshaped)
    local mu_reshaped = nn.Reshape(nMixture, 1, inputSize)(mu)
    local mu_pack = nn.SplitTable(2,4)(mu_reshaped)
    local pi_reshaped = nn.Reshape(nMixture, 1)(pi)
    local pi_pack = nn.SplitTable(2,3)(pi_reshaped)
    local target_reshaped = nn.Reshape(1, inputSize)(target)

    for i = 1, nMixture do
        local u_set = nn.SelectTable(i)(u_pack)
        local mu_set = nn.SelectTable(i)(mu_pack)
        local pi_set = nn.SelectTable(i)(pi_pack)

        local sigma = nn.CAddTable()({nn.MM()({nn.Transpose({2,3})(u_set), u_set}), eps})

        local det_sigma_2_pi = nn.Add(inputSize, inputSize * torch.log(2 * math.pi))
        (nn.LogDeterminant()(sigma))

        local sqr_det_sigma_2_pi = nn.MulConstant(-0.5)(det_sigma_2_pi)

        local target_mu = nn.CAddTable()({target_reshaped, nn.MulConstant(-1)(mu_set)})
        local transpose_target_mu = nn.Transpose({2,3})(target_mu)
        local inv_sigma = nn.Inverse()(sigma)
        local transpose_target_mu_sigma = 
        nn.MM()({target_mu, inv_sigma})
        local transpose_target_mu_sigma_target_mu = 
        nn.MM()({transpose_target_mu_sigma, transpose_target_mu})
        local exp_term = nn.MulConstant(-0.5)(transpose_target_mu_sigma_target_mu)

        local mixture_result = nn.CAddTable()({pi_set, sqr_det_sigma_2_pi, exp_term})

        if i == 1 then
            join_mixture_result = mixture_result
        else
            join_mixture_result = nn.JoinTable(2)({join_mixture_result, 
                mixture_result})
        end
    end
    
    -- Essentially this is the same a addlogsumexp   
    local max_mixture = nn.Max(2)(join_mixture_result)
    local max_expanded = nn.MulConstant(-1)(nn.Replicate(nMixture, 2, 2)(max_mixture))
    local norm_mixture = nn.CAddTable()({max_expanded, join_mixture_result})
    local norm_mixture_exp = nn.Exp()(norm_mixture)
    local norm_mixture_sumexp = nn.Sum(2)(norm_mixture_exp)
    local norm_mixture_logsumexp = nn.Log()(norm_mixture_sumexp)
    local norm_mixture_addlogsumexp = nn.CAddTable()({max_mixture, norm_mixture_logsumexp})
    norm_mixture_addlogsumexp = nn.MulConstant(-1)(norm_mixture_addlogsumexp)
    local result = nn.CMulTable()({mask, norm_mixture_addlogsumexp})

    return nn.gModule({pi, mu, u, mask, target, eps}, {result})
end

return mixture

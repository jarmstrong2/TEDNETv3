require 'cutorch'
require 'cunn'
require 'nn'
require 'nngraph'
require 'torch'
require 'logdeterminant'
require 'inverse'

local mixture = {}

function mixture.gauss(inputSize, uDimSize, nMixture)
    target = nn.Identity()()
    pi = nn.Identity()()
    mu = nn.Identity()()
    u = nn.Identity()()
    mask = nn.Identity()()
    eps = nn.Identity()()

    u_reshaped = nn.Reshape(nMixture, uDimSize, inputSize)(u)
    u_pack = nn.SplitTable(2,4)(u_reshaped)
    mu_reshaped = nn.Reshape(nMixture, 1, inputSize)(mu)
    mu_pack = nn.SplitTable(2,4)(mu_reshaped)
    pi_reshaped = nn.Reshape(nMixture, 1)(pi)
    pi_pack = nn.SplitTable(2,3)(pi_reshaped)
    target_reshaped = nn.Reshape(1, inputSize)(target)

    for i = 1, nMixture do
        u_set = nn.SelectTable(i)(u_pack)
        mu_set = nn.SelectTable(i)(mu_pack)
        pi_set = nn.SelectTable(i)(pi_pack)

        sigma = nn.CAddTable()({nn.MM()({nn.Transpose({2,3})(u_set), u_set}), eps})

        det_sigma_2_pi = nn.Add(inputSize, inputSize * torch.log(2 * math.pi))
        (nn.LogDeterminant()(sigma))

        sqr_det_sigma_2_pi = nn.MulConstant(-0.5)(det_sigma_2_pi)

        target_mu = nn.CAddTable()({target_reshaped, nn.MulConstant(-1)(mu_set)})
        transpose_target_mu = nn.Transpose({2,3})(target_mu)
        inv_sigma = nn.Inverse()(sigma)
        transpose_target_mu_sigma = 
        nn.MM()({target_mu, inv_sigma})
        transpose_target_mu_sigma_target_mu = 
        nn.MM()({transpose_target_mu_sigma, transpose_target_mu})
        exp_term = nn.MulConstant(-0.5)(transpose_target_mu_sigma_target_mu)

        mixture_result = nn.CAddTable()({pi_set, sqr_det_sigma_2_pi, exp_term})

        -- Essentially this is the same a addlogsumexp

        if i == 1 then
            join_mixture_result = mixture_result
        else
            join_mixture_result = nn.JoinTable(2)({join_mixture_result, 
                mixture_result})
        end

        max_mixture = nn.Max(2)(join_mixture_result)
        max_expanded = nn.MulConstant(-1)(nn.Replicate(nMixture, 2, 2)(max_mixture))
        norm_mixture = nn.CAddTable()({max_expanded, join_mixture_result})
        norm_mixture_exp = nn.Exp()(norm_mixture)
        norm_mixture_sumexp = nn.Sum(2)(norm_mixture_exp)
        norm_mixture_logsumexp = nn.Log()(norm_mixture_sumexp)
        norm_mixture_addlogsumexp = nn.CAddTable()({max_mixture, norm_mixture_logsumexp})
        norm_mixture_addlogsumexp = nn.MulConstant(-1)(norm_mixture_addlogsumexp)
        result = nn.CMulTable()({mask, norm_mixture_addlogsumexp})
    end

    return nn.gModule({pi, mu, u, mask, target, eps}, {result})
end

return mixture

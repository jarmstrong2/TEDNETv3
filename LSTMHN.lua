-- adapted from: wojciechz/learning_to_execute on github

local LSTMHN = {}

-- Creates one timestep of one LSTM
function LSTMHN.lstm(lookupSize, inputSize, hiddenSize)
    local lookup = nn.Identity()()
    local input = nn.Identity()()
    local w = nn.Identity()()
    local below_h = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms look-up vector
        local l2h            = nn.Linear(lookupSize, hiddenSize)(input)
        -- transforms input
        local i2h            = nn.Linear(inputSize, hiddenSize)(input)
        -- transforms window
        local w2h            = nn.Linear(32, hiddenSize)(w)
        -- transforms hidden output from below current hidden layer
        local bh2h            = nn.Linear(hiddenSize, hiddenSize)(below_h)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(hiddenSize, hiddenSize)(prev_h)
        return nn.CAddTable()({l2h, i2h, w2h, bh2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return nn.gModule({lookup, input, w, below_h, prev_c, prev_h}, {next_c, next_h})
end

return LSTMHN


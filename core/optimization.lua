--[[
	https://github.com/x-xxoa/synapsea
	core/optimization.lua

	MIT License
]]--

local syntable = require("core.syntable")
local optimization = {
	momentum
}

function optimization.momentum(args)
	for a, _ in pairs(args.gradient) do
		if type(args.gradient[a]) == "table" then
			args.gradient[a] = optimization.momentum{
				gradient = args.gradient[a],
				alpha = args.alpha,
				momentum = args.momentum,
				learningRate = args.learningRate
			}
		else
			args.momentum = args.momentum * args.alpha + args.gradient[a] * (1 - args.alpha)

			args.gradient[a] = args.gradient[a] + args.momentum * args.learningRate
		end
	end

	return args.gradient, {momentum = args.momentum}
end

return optimization

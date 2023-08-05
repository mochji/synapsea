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
	local lastGradient = 0

	for a, _ in pairs(args.gradient) do
		if type(args.gradient[a]) == "table" then
			args.gradient[a], lastGradient = optimization.momentum{
				gradient = args.gradient[a],
				alpha = args.alpha,
				momentum = args.momentum
			}
		else
			args.gradient[a] = args.momentum * args.lastGradient
		end
	end

	return args.gradient, lastGradient
end

return optimization

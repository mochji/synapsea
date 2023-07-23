--[[
	https://github.com/x-xxoa/synapsea
	core/optimization.lua

	MIT License

	TODO:

	add more optimizers
]]--

local syntable = require("core.syntable")
local optimization = {
	momentum
}

function optimization.momentum(args)
	for a, _ in pairs(args.gradient) do
		if type(args.gradient[a]) == "table" then
			args.gradient[a] = optimization.momentum(args.gradient[a], args.alpha, args.momentum, args.learningrate)
		else
			args.momentum = args.momentum * args.alpha + args.gradient[a]
			args.gradient[a] = args.gradient[a] + args.momentum * args.learningrate
		end
	end

	return {
		gradient = args.gradient,
		trainingParameters = {
			momentum = args.momentum
		}
	}
end

return optimization

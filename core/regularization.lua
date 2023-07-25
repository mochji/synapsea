--[[
	https://github.com/x-xxoa/synapsea
	core/regularization.lua

	MIT License
]]--

local syntable = require("core.syntable")
local regularization = {
	l1,
	l2
}

function regularization.l1(args)
	for a, _ in pairs(args.trainableParameters) do
		-- calculate l1 norm
		local l1Norm = syntable.absoluteSum({args.trainableParameters[a]}) -- putting the parameter in a table will ensure that absoluteSum will work even if its a number

		if type(args.gradient[a]) == "table" then
			args.gradient[a] = syntable.add(args.gradient[a], args.lambda * l1Norm)
		else
			args.gradient[a] = args.gradient[a] + args.lambda * l1Norm
		end
	end

	return args.gradient
end

function regularization.l2(args)
	for a, _ in pairs(args.trainableParameters) do
		-- calculate l2 norm
		local l2Norm = syntable.squaredSum({args.trainableParameters[a]}) -- the only reason syntable.squaredSum is a function is because of this function

		if type(args.gradient[a]) == "table" then
			args.gradient[a] = syntable.add(args.gradient[a], args.lambda * l2Norm)
		else
			args.gradient[a] = args.gradient[a] + args.lambda * l2Norm
		end
	end

	return args.gradient
end

return regularization

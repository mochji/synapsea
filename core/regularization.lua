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
		local l1norm = syntable.absoluteSum({args.trainableParameters[a]}) -- putting the parameter in a table will ensure that absoluteSum will work even if its a number

		if type(args.gradient[a]) == "table" then
			args.gradient[a] = syntable.add(args.gradient[a], args.lambda * l1norm)
		else
			args.gradient[a] = args.gradient[a] + args.lambda * l1norm
		end
	end

	return {
		gradient = args.gradient
	}
end

function regularization.l2(args)
	for a, _ in pairs(args.trainableParameters) do
		-- calculate l2 norm
		local l2norm = syntable.squaredSum({args.trainableParameters[a]})

		if type(args.gradient[a]) == "table" then
			args.gradient[a] = syntable.add(args.gradient[a], args.lambda * l2norm)
		else
			args.gradient[a] = args.gradient[a] + args.lambda * l2norm
		end
	end

	return {
		gradient = args.gradient
	}
end

return regularization

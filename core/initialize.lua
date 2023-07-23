--[[
	https://github.com/x-xxoa/synapsea
	core/initialize.lua

	MIT License
]]--

local synmath = require("core.synmath")
local initialize = {
	uniformRandom,
	normalRandom,
	uniformXavier,
	normalXavier,
	uniformHe,
	normalHe,
	constant
}

function initialize.uniformRandom(args)
	if type(args.input) ~= "table" then
		return synmath.random.uniform(args.lowerLimit, args.upperLimit)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.uniformRandom{
				input = args.input[a],
				lowerLimit = args.lowerLimit,
				upperLimit = args.upperLimit
			}
		else
			args.input[a] = synmath.random.uniform(args.lowerLimit, args.upperLimit)
		end
	end

	return args.input
end

function initialize.normalRandom(args)
	if type(args.input) ~= "table" then
		return synmath.random.normal(args.mean, args.sd)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.normalRandom{
				input = args.input[a],
				mean = args.mean,
				sd = args.sd
			}
		else
			args.input[a] = synmath.random.normal(args.mean, args.sd)
		end
	end

	return args.input
end

function initialize.uniformXavier(args)
	local limit = math.sqrt(6 / (args.inputs + args.outputs))

	if type(args.input) ~= "table" then
		return synmath.random.uniform(-limit, limit)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.uniformXavier{
				input = args.input[a],
				inputs = args.inputs,
				outputs = args.outputs
			}
		else
			args.input[a] = synmath.random.uniform(-limit, limit)
		end
	end

	return args.input
end

function initialize.normalXavier(args)
	local sd = math.sqrt(6 / (args.inputs + args.outputs))

	if type(args.input) ~= "table" then
		return synmath.random.normal(0, sd)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.normalXavier{
				input = args.input[a],
				inputs = args.inputs,
				outputs = args.outputs
			}
		else
			args.input[a] = synmath.random.normal(0, sd)
		end
	end

	return args.input
end

function initialize.uniformHe(args)
	local limit = math.sqrt(2 / args.inputs)

	if type(args.input) ~= "table" then
		return synmath.random.uniform(-limit, limit)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.uniformHe{
				input = args.input[a],
				inputs = args.inputs
			}
		else
			args.input[a] = synmath.random.uniform(-limit, limit)
		end
	end

	return args.input
end

function initialize.normalHe(args)
	local sd = math.sqrt(2 / args.inputs)

	if type(args.input) ~= "table" then
		return synmath.random.normal(0, sd)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.normalHe{
				input = args.input[a],
				inputs = args.inputs
			}
		else
			args.input[a] = synmath.random.normal(0, sd)
		end
	end

	return args.input
end

function initialize.constant(args)
	if type(args.input) ~= "table" then
		return args.value
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initialize.constant{
				input = args.input[a],
				value = args.value
			}
		else
			args.input[a] = args.value
		end
	end

	return args.input
end

return initialize

--[[
	https://github.com/x-xxoa/synapsea
	core/model.lua

	MIT License
]]--

local layerbuild = require("core.layerbuild")
local initialize = require("core.initialize")
local syntable = require("core.syntable")
local model = {}

function model.new(inputShape, metaData)
	return {
		inputShape = inputShape,
		metaData = metaData,
		layerConfig = {},
		trainingConfig = {}
	}
end

function model.addLayer(model, layerType, buildParameters)
	local layer = {
		type = layerType
	}

	if layerbuild[layerType] then

		if #model.layerConfig > 0 then
			buildParameters.inputShape = model.layerConfig[#model.layerConfig].outputShape
		else
			buildParameters.inputShape = model.inputShape
		end

		local build = layerbuild[layerType](buildParameters)

		for a, _ in pairs(build) do
			layer[a] = build[a]
		end
	else
		for a, _ in pairs(buildParameters) do
			layer[a] = buildParameters[a]
		end
	end

	model.layerConfig[#model.layerConfig + 1] = layer

	return model
end

function model.removeLayer(model,layerNumber)
	if not model.layerConfig[layerNumber] then
		error(string.format("Attempt to remove a non-existant layer. (%f/%d)", layerNumber, #model.layerConfig))
	end

	model.layerConfig[layerNumber] = nil

	return model
end

function model.initialize(args)
	for a = 1, #args.model.layerConfig do
		local layer = args.model.layerConfig[a]

		if layer.initializer then
			for b, initializer in pairs(layer.initializer) do
				local initParams = initializer.initializerParameters
				initParams.input = layer.parameters[b]

				layer.parameters[b] = initialize[initializer.initializer](initParams)

				initParams.input = nil
			end
		end
	end

	if args.optimizer then
		args.model.trainingConfig.optimizer = {
			optimizer = args.optimizer,
			parameters = args.optimizerParameters
		}
	end

	if args.regularizer then
		args.model.trainingConfig.regularizer = {
			regularizer = args.regularizer,
			parameters = args.regularizerParameters
		}
	end

	return args.model
end

function model.export(model, filename)
	local f = io.open(filename, "w")

	if not f then
		return false
	end

	f:write("return " .. syntable.toString(model))

	f:close()

	return true
end

--[[

-- model creation test code

local mod = model.new(
	{10},
	{
		name = "Example NN",
		author = "Katie",
		description = "A simple testing NN model :3"
	}
)

mod = model.addLayer(mod,
	"dense",
	{
		activation = "leakyrelu",
		alpha = 0.1,
		outputSize = 5,
		weightsInitializer = "normalRandom",
		weightsInitParameters = {
			mean = 0,
			sd = 0.1
		},
		weightsTrainable = true,
		biasInitializer = "constant",
		biasInitParameters = {
			value = 0.1
		},
		useBias = true,
		usePrelu = true
	}
)

mod = model.initialize{
	model = mod,
	optimizer = "momentum",
	optimizerParameters = {
		momentum = 0,
		alpha = 0.9
	}
}

print(syntable.toString(mod, true))
]]--

return model

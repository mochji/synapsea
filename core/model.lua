--[[
	https://github.com/x-xxoa/synapsea
	core/model.lua

	MIT License
]]--

local layerbuild = require("core.layerbuild")
local syntable = require("core.syntable")
local initialize = require("core.initialize")
local layer = require("core.layer")
local model = {}

function model.new(inputShape, metaData)
	local model = {
		metaData = metaData or {},
		inputShape = inputShape,
		parameterBuild = {},
		layerConfig = {},
		trainingConfig = {}
	}

	model.metaData.synapseaVersion = "v1.3.00-development"

	-- model functions

	model.addLayer = function(model, layerType, buildParameters, layerNumber)
		buildParameters, layerNumber = buildParameters or {}, layerNumber or #model.layerConfig + 1

		local layer = {}

		if layerbuild[layerType] then
			-- get last layer output shape

			if #model.layerConfig == 0 then
				buildParameters.inputShape = model.inputShape
			else
				buildParameters.inputShape = model.layerConfig[layerNumber - 1].outputShape
			end

			layer, parameterBuild = layerbuild[layerType](buildParameters)
			table.insert(model.parameterBuild, layerNumber, parameterBuild or {})
		else
			layer = buildParameters
		end

		layer.type = layerType
		table.insert(model.layerConfig, layerNumber, layer)
	end

	model.removeLayer = function(model, layerNumber)
		layerNumber = layerNumber or #model.layerConfig

		assert(model.layerConfig[layerNumber], string.format("Attempt to remove a non-existant layer (%d)", layerNumber))

		table.remove(model.layerConfig[layerNumber])
		table.remove(model.parameterBuild[layerNumber])
	end

	model.export = function(model, fileName, format)
		local f, err, code = io.open(fileName, "w")

		assert(f, string.format("Couldn't open %s for writing, %s (%s)", fileName, err, code))

		f:write("return " .. syntable.toString(model, format, false))
		f:close()
	end

	model.summary = function(model, returnString)
		-- get needed variables

		local function listToString(list)
			local listStr = "["

			for a = 1, #list do
				listStr = listStr .. list[a]

				if a < #list then
					listStr = listStr .. ", "
				end
			end

			return listStr .. "]"
		end

		-- calculate how much spacing to use

		local layerNameSizes = {}

		for a = 1, #model.layerConfig do
			layerNameSizes[a] = #model.layerConfig[a].type
		end

		local layerSpacing = math.max(math.max(table.unpack(layerNameSizes)) + 1, 16)
		local totalLength = layerSpacing + 78

		-- get total amount of parameters

		local parameterCount = {
			trainable = 0,
			nonTrainable = 0
		}

		for a = 1, #model.layerConfig do
			local layer = model.layerConfig[a]
			parameterCount[a] = {
				trainable = 0,
				nonTrainable = 0
			}

			-- get parameters in layer.parameters

			if layer.parameters then
				for parameterName, parameter in pairs(layer.parameters) do
					if layer.trainable[parameterName] then
						if type(parameter) ~= "table" then
							parameterCount.trainable = parameterCount.trainable + 1
							parameterCount[a].trainable = parameterCount[a].trainable + 1
						else
							local parametersInLayer = syntable.totalItems(parameter)

							parameterCount.trainable = parameterCount.trainable + parametersInLayer
							parameterCount[a].trainable = parameterCount[a].trainable + parametersInLayer
						end
					else
						if type(parameter) ~= "table" then
							parameterCount.nonTrainable = parameterCount.nonTrainable + 1
							parameterCount[a].nonTrainable = parameterCount[a].nonTrainable + 1
						else
							local parametersInLayer = syntable.totalItems(parameter)

							parameterCount.nonTrainable = parameterCount.nonTrainable + parametersInLayer
							parameterCount[a].nonTrainable = parameterCount[a].nonTrainable + parametersInLayer
						end
					end
				end
			end

			-- get parameters in model.parameterBuild

			if model.parameterBuild then
				local build = model.parameterBuild[a]

				for parameterName, parameter in pairs(build) do
					local parametersInLayer = syntable.product(parameter.shape)

					if layer.trainable[parameterName] then
						parameterCount.trainable = parameterCount.trainable + parametersInLayer
						parameterCount[a].trainable = parameterCount[a].trainable + parametersInLayer
					else
						parameterCount.nonTrainable = parameterCount.nonTrainable + parametersInLayer
						parameterCount[a].nonTrainable = parameterCount.nonTrainable + parametersInLayer
					end
				end
			end
		end

		-- make the summary string

		local summaryStr = "Model\n"

		if model.parameterBuild then
			summaryStr = "Uninitialized Model\n"
		end

		-- layer data

		summaryStr = string.format("%s%s\nLayer Type%sInput Shape     Output Shape    Layer #         # Of Parameters\n%s\n", summaryStr, string.rep("=", totalLength), string.rep(" ", layerSpacing - 10), string.rep("=", totalLength))

		for a = 1, #model.layerConfig do
			local layer = model.layerConfig[a]
			local inputShape, outputShape = listToString(layer.inputShape), listToString(layer.outputShape)

			summaryStr = string.format("%s%s%s%s%s%s%s%d%s%d\n", summaryStr, layer.type, string.rep(" ", layerSpacing - layerNameSizes[a]), inputShape, string.rep(" ", 16 - #inputShape), outputShape, string.rep(" ", 16 - #outputShape), a, string.rep(" ", 16 - #tostring(a)), parameterCount[a].trainable + parameterCount[a].nonTrainable)

			if a < #model.layerConfig then
				summaryStr = summaryStr .. string.rep("-", totalLength) .. "\n"
			end
		end

		summaryStr = summaryStr .. string.rep("=", totalLength) .. "\n"

		-- global data

		summaryStr = string.format("%s\nTotal Layers: %d\nHidden Layers: %d\n\nTotal Parameters: %d\nTrainable Parameters: %d\nNon-Trainable Parameters: %d\nSynapsea Version: %s\n", summaryStr, #model.layerConfig + 2, #model.layerConfig, parameterCount.trainable + parameterCount.nonTrainable, parameterCount.trainable, parameterCount.nonTrainable, model.metaData.synapseaVersion)

		if returnString then
			return summaryStr
		end

		io.write(summaryStr)
	end

	model.initialize = function(model, optimizer, optimizerParameters, regularizer, regularizerParameters)
		-- create parameters in layers and initialize

		for a = 1, #model.parameterBuild do
			local layer = model.layerConfig[a]

			for parameterName, parameter in pairs(model.parameterBuild[a]) do
				layer.parameters[parameterName] = syntable.new(parameter.shape, 0)

				if layer.initializer[parameterName] then
					layer.initializer[parameterName].parameters.input = layer.parameters[parameterName]
					layer.parameters[parameterName] = initialize[layer.initializer[parameterName].initializer](layer.initializer[parameterName].parameters)
					layer.initializer[parameterName].parameters.input = nil
				end
			end
		end

		-- create training parameters

		if optimizer then
			model.trainingConfig.optimizer = {
				optimizer = optimizer,
				parameters = optimizerParameters
			}
		end

		if regularizer then
			model.trainingConfig.regularizer = {
				regularizer = regularizer,
				parameters = regularizerParameters
			}
		end

		model.addLayer, model.removeLayer, model.parameterBuild, model.initialize = nil, nil, nil, nil
		model.outputShape = model.layerConfig[#model.layerConfig].outputShape
		model.layers = #model.layerConfig

		model.forwardPass = function(model, input)
			local function layerToParameters(layer)
				local parameters = {}

				if layer.parameters then
					for parameter, _ in pairs(layer.parameters) do
						parameters[parameter] = layer.parameters[parameter]
					end
				end

				if layer.config then
					for config, _ in pairs(layer.config) do
						parameters[config] = layer.config[config]
					end
				end

				return parameters
			end

			local lastOutput = input

			for a = 1, #model.layerConfig do
				local parameters = layerToParameters(model.layerConfig[a])
				parameters.input = lastOutput

				lastOutput = layer[model.layerConfig[a].type](parameters)
			end

			return lastOutput
		end
	end

	return model
end

return model

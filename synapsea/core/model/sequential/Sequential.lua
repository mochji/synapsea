--[[
	https://github.com/mochji/synapsea
	core/model/sequential/Sequential.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
	Copyright (C) 2023 mochji

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
]]--

local canindex           = require("core.utils.canindex")

local layersModule       = require("core.layers.layers")
local buildModule        = require("core.layers.build")
local initializersModule = require("core.initializers")

local Sequential = {
	add,
	pop,
	build,
	summary,
	export,
	import,
	fit,
	forwardPass
}

function Sequential.add(model, layerType, buildParameters)
	assert(
		layerType,
		"bad argument #1 to 'add' (layer type expected)"
	)

	buildParameters = buildParameters or {}

	local layerNumber = #model.layerConfig + 1

	local layer, parameterBuild

	if buildModule[layerType] then
		if layerNumber == 1 then
			buildParameters.inputShape = model.inputShape
		else
			buildParameters.inputShape = model.layerConfig[layerNumber - 1].outputShape
		end

		layer, parameterBuild = buildModule[layerType](buildParameters)
	else
		layer = buildParameters
	end

	layer.type = layerType

	model.layerConfig[layerNumber]    = layer
	model.parameterBuild[layerNumber] = parameterBuild or {}

	return model
end

function Sequential.pop(model)
	assert(
		#model.layerConfig > 0,
		"attempt to pop model with no layers"
	)

	model.layerConfig[#model.layerConfig] = nil

	if model.parameterBuild then
		model.parameterBuild[#model.parameterBuild] = nil
	end

	return model
end

function Sequential.addTrainingConfig(model, args)
	args = args or {}

	if args.optimizer then
		model.trainingConfig.optimizer = {
			optimizer = args.optimizer,
			args      = args.optimizerArgs
		}
	end

	if args.regularizer then
		model.trainingConfig.regularizer = {
			regularizer = args.regularizer,
			args        = args.regularizerArgs
		}
	end

	model.trainingConfig.learningRate =
		args.learningRate or model.trainingConfig.learningRate

	model.trainingConfig.epochs =
		args.epochs or model.trainingConfig.epochs

	model.trainingConfig.loss =
		args.loss or model.trainingConfig.loss

	return model
end

function Sequential.build(model)
	assert(
		#model.layerConfig > 0,
		"attempt to initialize model with no layers"
	)

	if model.parameterBuild then
		for a = 1, #model.parameterBuild do
			local layer = model.layerConfig[a]

			for parameterName, parameter in pairs(model.parameterBuild[a]) do
				layer.parameters[parameterName] = initializersModule[layer.initializer[parameterName].initializer](
					model.parameterBuild[a][parameterName],
					layer.initializer[parameterName].args
				)
			end
		end
	end

	model.parameterBuild = nil
	model.add            = nil

	model.forwardPass    = Sequential.forwardPass
	model.fit            = Sequential.fit

	return model
end

function Sequential.summary(model, returnString)
end

function Sequential.export(model, fileName)
	local tableToString

	tableToString = function(table)
		local output = "{"

		for i, v in pairs(table) do
			local valueType = type(v)

			if valueType ~= "function" then
				if type(i) == "number" then
					output = output .. string.format("[%s]", i) .. "="
				else
					output = output .. string.format("[%q]", i) .. "="
				end

				if canindex(v) then
					output = output .. tableToString(v)
				elseif valueType == "string" then
					output = output .. string.format("%q", v)
				elseif valueType ~= valueType then
					output = output .. "0/0"
				elseif valueType == 2^1024 then
					output = output .. "2^1024"
				else
					output = output .. tostring(v)
				end

				output = output .. ","
			end
		end

		if output:sub(#output) == "," then
			output = output:sub(1, #output - 1)
		end

		return output .. "}"
	end

	local f, err = io.open(fileName, "w")

	assert(f, fileName .. ": " .. (err or "nil error"))

	f:write("return " .. tableToString(model))

	f:close()
end

function Sequential.import(fileName)
	local model = dofile(fileName)

	if not model then
		return nil
	end

	if model.parameterBuild then
		model.add = Sequential.add
	else
		model.fit         = Sequential.fit
		model.forwardPass = Sequential.forwardPass
	end

	model.pop     = Sequential.pop
	model.build   = Sequential.build
	model.summary = Sequential.summary

	if model.metaData.synapseaVersion:match("development") then
		io.write("\27[1m\27[33mWARNING:\27[0m This model was created in a development release of Synapsea!\n")
		io.flush()
	end

	return model
end

function Sequential.fit(model)
end

function Sequential.forwardPass(model, input)
	local output = input

	for a = 1, #model.layerConfig do
		local layer = model.layerConfig[a]

		local args = {}

		if layer.parameters then
			for parameterName, parameter in pairs(layer.parameters) do
				args[parameterName] = parameter
			end
		end

		if layer.config then
			for configName, config in pairs(layer.config) do
				args[configName] = config
			end
		end

		args.input = output

		output = layersModule[model.layerConfig[a].type](args)
	end

	return output
end

return setmetatable(
	Sequential,
	{
		__call = function(_, inputShape, metaData)
			local model = {
				metaData = metaData or {},
				inputShape = inputShape,
				parameterBuild = {},
				layerConfig = {},
				trainingConfig = {}
			}

			assert(inputShape, "bad argument #1 to 'Sequential' (input shape expected)")

			model.metaData.synapseaVersion = "v2.0.00-development"

			model.add               = Sequential.add
			model.pop               = Sequential.pop
			model.build             = Sequential.build
			model.addTrainingConfig = Sequential.addTrainingConfig
			model.summary           = Sequential.summary
			model.export            = Sequential.export

			return model
		end
	}
)

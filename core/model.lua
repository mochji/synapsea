--[[
	https://github.com/mochji/synapsea
	core/model.lua

	Synapsea, simple yet powerful machine learning platform for Lua.
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

local canindex           = require("core.canindex")

local layersModule       = require("core.layers.layers")
local buildModule        = require("core.layers.build")
local initializersModule = require("core.initializers")
local backPropModule     = require("core.backprop")

local modelModule = {
	new,
	add,
	pop,
	initialize,
	summary,
	export,
	import,
	fit,
	forwardPass
}

function modelModule.new(inputShape, metaData)
	local model = {
		metaData = metaData or {},
		inputShape = inputShape,
		parameterBuild = {},
		layerConfig = {},
		trainingConfig = {}
	}

	model.metaData.synapseaVersion = "v2.0.00-unstable"

	model.add        = modelModule.add
	model.pop        = modelModule.pop
	model.initialize = modelModule.initialize
	model.summary    = modelModule.summary
	model.export     = modelModule.export

	return model
end

function modelModule.add(model, layerType, buildParameters)
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

function modelModule.pop(model)
	assert(
		#model.layerConfig >= 1,
		"attempt to pop model with no layers"
	)

	model.layerConfig[#model.layerConfig] = nil

	if model.parameterBuild then
		model.parameterBuild[#model.parameterBuild] = nil
	end

	return model
end

function modelModule.initialize(model, args)
	if model.parameterBuild then
		for a = 1, #model.parameterBuild do
			local layer = model.layerConfig[a]

			for parameterName, parameter in pairs(model.parameterBuild[a]) do
				layer.parameters[parameterName] = initializersModule[layer.initializer[parameterName].initializer](
					model.parameterBuild[a][parameterName].shape,
					layer.initializer[parameterName].parameters
				)
			end
		end
	end

	if args.optimizer then
		model.trainingConfig.optimizer = {
			optimizer = args.optimizer,
			parameters = args.optimizerParameters
		}
	end

	if args.regularizer then
		model.trainingConfig.regularizer = {
			regularizer = args.regularizer,
			parameters = args.regularizerParameters
		}
	end

	-- This is done this way to avoid overwriting training data when initializing after first initialization
	-- to re-initialize parameters (to make a specific edge case easier)

	if args.learningRate then
		model.trainingConfig.learningRate = args.learningRate
	end

	if args.epochs then
		model.trainingConfig.epochs = args.epochs
	end

	if args.loss then
		model.trainingConfig.loss = args.loss
	end

	if not model.layerConfig[#model.layerConfig] then
		model.outputShape = model.inputShape
	else
		model.outputShape = model.layerConfig[#model.layerConfig].outputShape
	end

	model.parameterBuild = nil
	model.add            = nil

	model.forwardPass    = modelModule.forwardPass
	model.fit            = modelModule.fit

	return model
end

function modelModule.summary(model, returnString)
end

function modelModule.export(model, fileName)
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

function modelModule.import(fileName)
	local model = dofile(fileName)

	if not model then
		return nil
	end

	if model.parameterBuild then
		model.add = modelModule.add
	else
		model.fit         = modelModule.fit
		model.forwardPass = modelModule.forwardPass
	end

	model.pop        = modelModule.pop
	model.initialize = modelModule.initialize
	model.summary    = modelModule.summary

	return model
end

function modelModule.fit(model, algorithm, dataset, args)
	return backPropModule.gradientDescent[algorithm](model, dataset, args)
end

function modelModule.forwardPass(model, input)
	local output = input

	for a = 1, #model.layerConfig do
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

return modelModule

--[[
	Synapsea v1.3.00-unstable

	A Lua Neural Network library made in pure Lua.

	Read the README.md file for documentation and information, 

	https://github.com/x-xxoa/synapsea

	MIT License
]]--

--import the core files

local syn = {
	version = "v1.3.00-unstable",
	table = require("core.syntable"),
	model = require("core.model"),
	initialize = require("core.initialize")
}

local model = syn.model.new(
	{64, 64},
	{
		this = "help my lego movie phase",
		is = "coming back because of one",
		metadata = "tiktok i saw help please"
	}
)

model:addLayer(
	"averagePooling2D",
	{
		kernel = {6, 6},
		stride = {2, 2}
	}
)

model:addLayer(
	"convolutional2D",
	{
		activation = "relu",
		kernel = {6, 6},
		stride = {2, 2},
		filterInitializer = "normalRandom",
		filterInitParameters = {
			sd = 1,
			mean = 0
		},
		biasInitializer = "constant",
		biasInitParameters = {
			value = 0.1
		},
		filterTrainable = true,
		useBias = true
	}
)

model:addLayer(
	"averagePooling2D",
	{
		kernel = {2, 2},
		stride = {2, 2}
	}
)

model:addLayer("flatten")

model:addLayer(
	"dense",
	{
		activation = "leakyrelu",
		alpha = 0.1,
		outputSize = 5,
		weightsInitializer = "uniformRandom",
		weightsInitParameters = {
			lowerLimit = -1,
			upperLimit = 1
		},
		weightsTrainable = true,
		usePrelu = true
	}
)

model:export("modeltest.lua", true)

model:initialize(
	"momentum",
	{momentum = 0.9}
)

model:summary()

local input = syn.table.new({64, 64}, 0)

input = syn.initialize.uniformRandom{
	input = input,
	lowerLimit = 0,
	upperLimit = 1
}

for i,v in pairs(model:forwardPass(input)) do print(i,v) end

return syn

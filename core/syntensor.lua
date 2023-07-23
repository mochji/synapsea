--[[
	https://github.com/x-xxoa/synapsea
	core/syntensor.lua

	MIT License
]]--

local syntable = require("syntable")
local syntensor = {}

function syntensor.new(dimensions, defaultvalue, index)
	index = index or 1 -- if index is nil it will be set to 1

	local tensor = {}

	if index == 1 then
		tensor.dimensions = dimensions
	end

	if index == #dimensions then
		for a = 1, dimensions[index] do
			tensor[a] = defaultvalue
		end
	else
		for a = 1, dimensions[index] do
			tensor[a] = syntensor.new(dimensions, defaultvalue, index + 1)
		end
	end

	return tensor
end

function syntensor.flatten(tensor)
	if #tensor.dimensions == 1 then
		return tensor
	end

	local flattenedTensor = {}

	for a = 1, #tensor do
		
function syntensor.reshape()
end

function syntensor.transpose()
end

function syntensor.concatenate()
end

function syntensor.split()
end

local ten = tensor.new({3,4}, 0)

print(#ten)

print(syntable.toString(ten, true))

return tensor

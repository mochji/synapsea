--[[
	https://github.com/x-xxoa/synapsea
	core/syntensor.lua

	MIT License
]]--

local syntensor = {}

function syntensor.new(dimensions, defaultValue, index)
	index = index or 1 -- if index is nil it will be set to 1

	local tensor = {}

	if index == 1 then
		tensor.dimensions = dimensions
	end

	if index == #dimensions then
		for a = 1, dimensions[index] do
			tensor[a] = defaultValue
		end
	else
		for a = 1, dimensions[index] do
			tensor[a] = syntensor.new(dimensions, defaultValue, index + 1)
		end
	end

	return tensor
end

function syntensor.flatten(tensor)
end

return syntensor

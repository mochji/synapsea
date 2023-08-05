--[[
	https://github.com/x-xxoa/synapsea
	core/synmath.lua

	MIT License
]]--

local synmath = {
	random = {
		uniform,
		normal
	},
	sign,
	cbrt,
	root
}

function synmath.random.uniform(lowerLimit, upperLimit)
	return lowerLimit + math.random() * (upperLimit - lowerLimit) -- random float between upperLimit and lowerLimit
end

function synmath.random.normal(mean, sd)
	return mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd -- https://forum.cheatengine.org/viewtopic.php?p=5724230 old forums have pretty much everything lol
end

function synmath.round(x)
	return math.floor(x + 0.5)
end

function synmath.sign(x)
	if x > 0 then
		return 1
	end

	if x < 0 then
		return -1
	end

	return 0
end

function synmath.root(x, root)
	return x^(1 / root)
end

return synmath

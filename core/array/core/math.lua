--[[
	https://github.com/x-xxoa/synapsea
	core/array/core/math.lua

	Synapsea, a machine learning library made in pure Lua.
	Copyright (C) 2023 x-xxoa
																		   
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

local mathModule = {
	addRandom = {
		uniform,
		normal
	},
	subtractRandom = {
		uniform,
		normal
	},
	multiplyRandom = {
		uniform,
		normal
	},
	divideRandom = {
		uniform,
		normal
	},
	add,
	subtract,
	multiply,
	divide,
	negate,
	modulo,
	power,
	floorDivide,
	sum,
	absoluteSum,
	exponentSum,
	difference,
	absoluteDifference,
	exponentDifference,
	product,
	absoluteProduct,
	exponentProduct,
	quotient,
	absoluteQuotient,
	exponentQuotient,
	max,
	min,
	lessThan,
	lessThanOrEqualTo,
	greaterThan,
	greaterThanOrEqualTo,
	absolute,
	ceil,
	floor,
	round,
	sin,
	cos,
	exp,
	rad,
	sqrt
}

function mathModule.addRandom.uniform(tbl, lowerLimit, upperLimit)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.addRandom.normal(tbl[a], lowerLimit, upperLimit)
		else
			tbl[a] = tbl[a] + lowerLimit + math.random() * (upperLimit - lowerLimit)
		end
	end

	return tbl
end

function mathModule.addRandom.normal(tbl, mean, sd)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.addRandom.uniform(tbl[a], mean, sd)
		else
			tbl[a] = tbl[a] + (mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd)
		end
	end

	return tbl
end

function mathModule.subtractRandom.uniform(tbl, lowerLimit, upperLimit)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.subtractRandom.normal(tbl[a], lowerLimit, upperLimit)
		else
			tbl[a] = tbl[a] + lowerLimit + math.random() * (upperLimit - lowerLimit)
		end
	end

	return tbl
end

function mathModule.subtractRandom.normal(tbl, mean, sd)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.subtractRandom.uniform(tbl[a], mean, sd)
		else
			tbl[a] = tbl[a] + (mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd)
		end
	end

	return tbl
end

function mathModule.multiplyRandom.uniform(tbl, lowerLimit, upperLimit)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.multiplyRandom.normal(tbl[a], lowerLimit, upperLimit)
		else
			tbl[a] = tbl[a] * (lowerLimit + math.random() * (upperLimit - lowerLimit))
		end
	end

	return tbl
end

function mathModule.multiplyRandom.normal(tbl, mean, sd)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.addRandom.uniform(tbl[a], mean, sd)
		else
			tbl[a] = tbl[a] * (mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd)
		end
	end

	return tbl
end

function mathModule.divideRandom.uniform(tbl, lowerLimit, upperLimit)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.multiplyRandom.normal(tbl[a], lowerLimit, upperLimit)
		else
			tbl[a] = tbl[a] / (lowerLimit + math.random() * (upperLimit - lowerLimit))
		end
	end

	return tbl
end

function mathModule.divideRandom.normal(tbl, mean, sd)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.addRandom.uniform(tbl[a], mean, sd)
		else
			tbl[a] = tbl[a] / (mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd)
		end
	end

	return tbl
end

function mathModule.add(a, b)
	if type(a) == "number" then
		local temp = a
		a = b
		b = temp
	end

	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.add(a[i], b)
			else
				a[i] = a[i] + b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.add(a[i], b[i])
		else
			a[i] = a[i] + b[i]
		end
	end

	return a
end

function mathModule.subtract(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.subtract(a[i], b)
			else
				a[i] = a[i] - b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.subtract(a[i], b[i])
		else
			a[i] = a[i] - b[i]
		end
	end

	return a
end

function mathModule.multiply(a, b)
	if type(a) == "number" then
		local temp = a
		a = b
		b = temp
	end

	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.multiply(a[i], b)
			else
				a[i] = a[i] * b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.multiply(a[i], b[i])
		else
			a[i] = a[i] * b[i]
		end
	end

	return a
end

function mathModule.divide(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.divide(a[i], b)
			else
				a[i] = a[i] / b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.divide(a[i], b[i])
		else
			a[i] = a[i] / b[i]
		end
	end

	return a
end

function mathModule.negate(a)
	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.negate(a[i])
		else
			a[i] = -a[i]
		end
	end

	return a
end

function mathModule.modulo(a, b)
	if type(a) == "number" then
		local temp = a
		a = b
		b = temp
	end

	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.modulo(a[i], b)
			else
				a[i] = a[i] % b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.modulo(a[i], b[i])
		else
			a[i] = a[i] % b[i]
		end
	end

	return a
end

function mathModule.power(a, b)
	if type(a) == "number" then
		local temp = a
		a = b
		b = temp
	end

	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.power(a[i], b)
			else
				a[i] = a[i]^b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.power(a[i], b[i])
		else
			a[i] = a[i]^b[i]
		end
	end

	return a
end

function mathModule.floorDivide(a, b)
	if type(a) == "number" then
		local temp = a
		a = b
		b = temp
	end

	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				a[i] = mathModule.floorDivide(a[i], b)
			else
				a[i] = math.floor(a[i] / b)
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			a[i] = mathModule.floorDivide(a[i], b[i])
		else
			a[i] = math.floor(a[i] / b[i])
		end
	end

	return a
end

function mathModule.sum(tbl)
	local sum = 0

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			sum = sum + mathModule.sum(tbl[a])
		else
			sum = sum + a[i]
		end
	end

	return sum
end

function mathModule.absoluteSum(tbl)
	local sum = 0

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			sum = sum + mathModule.absoluteSum(tbl[a])
		else
			sum = sum + math.abs(a[i])
		end
	end

	return sum
end

function mathModule.exponentSum(tbl, exponent)
	local sum = 0

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			sum = sum + mathModule.exponentSum(tbl[a], exponent)
		else
			sum = sum + a[i]^exponent
		end
	end

	return sum
end

function mathModule.product(tbl)
	local product = 1

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			product = product * mathModule.product(tbl[a])
		else
			product = product * a[i]
		end
	end

	return sum
end

function mathModule.absoluteProduct(tbl)
	local product = 1

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			product = product * mathModule.absoluteProduct(tbl[a])
		else
			product = product * math.abs(tbl[a])
		end
	end

	return product
end

function mathModule.exponentProduct(tbl, exponent)
	local product = 1

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			product = product * mathModule.exponentProduct(tbl[a], exponent)
		else
			product = product * tbl[a]^exponent
		end
	end

	return product
end

function mathModule.difference(tbl)
	local difference = next(tbl)
	local isFirstItem = true

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			difference = difference - mathModule.difference(tbl[a])
		else
			if not isFirstItem then
				difference = difference - a[i]
			else
				isFirstItem = false
			end
		end
	end

	return difference
end

function mathModule.absoluteDifference(tbl)
	local difference = math.abs(next(tbl))
	local isFirstItem = true

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			difference = difference - mathModule.absoluteDifference(tbl[a])
		else
			if not isFirstItem then
				difference = difference - math.abs(a[i])
			else
				isFirstItem = false
			end
		end
	end

	return difference
end

function mathModule.exponentDifference(tbl, exponent)
	local difference = next(tbl)^exponent
	local isFirstItem = true

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			difference = difference - mathModule.exponentDifference(tbl[a], exponent)
		else
			if not isFirstItem then
				difference = difference - a[i]^exponent
			else
				isFirstItem = false
			end
		end
	end

	return difference
end

function mathModule.quotient(tbl)
	local quotient = next(tbl)
	local isFirstItem = true

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			quotient = quotient / mathModule.absoluteQuotient(tbl[a])
		else
			if not isFirstItem then
				quotient = quotient / a[i]
			else
				isFirstItem = false
			end
		end
	end

	return quotient
end

function mathModule.absoluteQuotient(tbl)
	local quotient = math.abs(next(tbl))
	local isFirstItem = true

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			quotient = quotient / mathModule.absoluteQuotient(tbl[a])
		else
			if not isFirstItem then
				quotient = quotient / math.abs(a[i])
			else
				isFirstItem = false
			end
		end
	end

	return quotient
end

function mathModule.exponentQuotient(tbl, exponent)
	local quotient = next[i]^exponent
	local isFirstItem = true

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			quotient = quotient / mathModule.exponentQuotient(tbl[a], exponent)
		else
			if not isFirstItem then
				quotient = quotient^a[i]
			else
				isFirstItem = false
			end
		end
	end

	return quotient
end

function mathModule.max(tbl)
	local max

	for a = 1, #tbl do
		if type(tbl) == "table" then
			if not max then
				max = mathModule.max(tbl[a])
			else
				max = math.max(max, mathModule.max(tbl[a]))
			end
		else
			if not max then
				max = tbl[a]
			else
				max = math.max(max, tbl[a])
			end
		end
	end

	return max
end

function mathModule.min(tbl)
	local min

	for a = 1, #tbl do
		if type(tbl) == "table" then
			if not min then
				min = mathModule.min(tbl[a])
			else
				min = math.min(min, mathModule.min(tbl[a]))
			end
		else
			if not min then
				min = tbl[a]
			else
				min = math.min(min, tbl[a])
			end
		end
	end

	return min
end

function mathModule.lessThan(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				if not mathModule.lessThan(a[i], b) then
					return false
				end
			else
				if a[i] >= b then
					return false
				end
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			if not mathModule.lessThan(a[i], b[i]) then
				return false
			end
		else
			if a[i] >= b[i] then
				return false
			end
		end
	end

	return true
end

function mathModule.lessThanOrEqualTo(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				if not mathModule.lessThanOrEqualTo(a[i], b) then
					return false
				end
			else
				if a[i] > b then
					return false
				end
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			if not mathModule.lessThanOrEqualTo(a[i], b[i]) then
				return false
			end
		else
			if a[i] > b[i] then
				return false
			end
		end
	end

	return true
end

function mathModule.greatherThan(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				if not mathModule.greatherThan(a[i], b) then
					return false
				end
			else
				if a[i] <= b then
					return false
				end
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			if not mathModule.greatherThan(a[i], b[i]) then
				return false
			end
		else
			if a[i] <= b[i] then
				return false
			end
		end
	end

	return true
end

function mathModule.greatherThanOrEqualTo(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "table" then
				if not mathModule.greatherThanOrEqualTo(a[i], b) then
					return false
				end
			else
				if a[i] < b then
					return false
				end
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "table" then
			if not mathModule.greatherThanOrEqualTo(a[i], b[i]) then
				return false
			end
		else
			if a[i] < b[i] then
				return false
			end
		end
	end

	return true
end

function mathModule.absolute(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.absolute(tbl[a])
		else
			tbl[a] = math.abs(tbl[a])
		end
	end

	return tbl
end

function mathModule.ceil(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.ceil(tbl[a])
		else
			tbl[a] = math.ceil(tbl[a])
		end
	end

	return tbl
end

function mathModule.floor(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.floor(tbl[a])
		else
			tbl[a] = math.floor(tbl[a])
		end
	end

	return tbl
end

function mathModule.round(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.round(tbl[a])
		else
			tbl[a] = math.floor(tbl[a] + 0.5)
		end
	end

	return tbl
end

function mathModule.sin(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.sin(tbl[a])
		else
			tbl[a] = math.sin(tbl[a])
		end
	end

	return tbl
end

function mathModule.cos(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.cos(tbl[a])
		else
			tbl[a] = math.cos(tbl[a])
		end
	end

	return tbl
end

function mathModule.exp(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.exp(tbl[a])
		else
			tbl[a] = math.exp(tbl[a])
		end
	end

	return tbl
end

function mathModule.rad(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.rad(tbl[a])
		else
			tbl[a] = math.rad(tbl[a])
		end
	end

	return tbl
end

function mathModule.sqrt(tbl)
	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			tbl[a] = mathModule.rad(tbl[a])
		else
			tbl[a] = math.rad(tbl[a])
		end
	end

	return tbl
end

return mathModule

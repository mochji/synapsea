--[[
	https://github.com/x-xxoa/synapsea
	core/syntable.lua

	MIT License
]]--

local syntable = {
	flatten,
	sum,
	absoluteSum,
	squaredSum,
	difference,
	absoluteDifference,
	product,
	absoluteProduct,
	quotient,
	absoluteQuotient,
	max,
	min,
	add,
	multiply,
	divide,
	find,
	new,
	toString
}

function syntable.flatten(table)
	local flattenedTable = {}

	for a = 1, #table do
		if type(table[a]) == "table" then
			local returnTable = syntable.flatten(table[a])

			for b = 1, #returnTable do
				flattenedTable[#flattenedTable + 1] = returnTable[b]
			end
		else
			flattenedTable[#flattenedTable + 1] = table[a]
		end
	end

	return flattenedTable
end

function syntable.sum(table)
	local sum = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			sum = sum + syntable.sum(table[a])
		else
			sum = sum + table[a]
		end
	end

	return sum
end

function syntable.absoluteSum(table)
	local sum = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			sum = sum + syntable.absoluteSum(table[a])
		else
			sum = sum + math.abs(table[a])
		end
	end

	return sum
end

function syntable.squaredSum(table)
	local sum = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			sum = sum + syntable.difference(table[a])
		else
			sum = sum + table[a]^2
		end
	end

	return sum
end

function syntable.difference(table)
	local difference = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			difference = difference - syntable.difference(table[a])
		else
			difference = difference - table[a]
		end
	end

	return difference
end

function syntable.absoluteDifference(table)
	local difference = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			difference = difference - syntable.difference(table[a])
		else
			difference = difference - math.abs(table[a])
		end
	end

	return difference
end

function syntable.product(table)
	local product = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			product = product * syntable.product(table[a])
		else
			product = product * table[a]
		end
	end

	return product
end

function syntable.absoluteProduct(table)
	local product = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			product = product * syntable.absoluteProduct(table[a])
		else
			product = product * math.abs(table[a])
		end
	end

	return product
end

function syntable.quotient(table)
	local quotient = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			quotient = quotient / syntable.quotient(table[a])
		else
			quotient = quotient / table[a]
		end
	end

	return quotient
end

function syntable.absoluteQuotient(table)
	local quotient = 0

	for a = 1, #table do
		if type(table[a]) == "table" then
			quotient = quotient / syntable.absoluteQuotient(table[a])
		else
			quotient = quotient / math.abs(table[a])
		end
	end

	return quotient
end

function syntable.max(table)
	local max

	for a = 1, #table do
		if type(table[a]) == "table" then
			local returnedMax = syntable.max(table[a])

			if not max or returnedMax > max then
				max = returnedMax
			end
		else
			if not max or table[i] > max then
				max = table[a]
			end
		end
	end

	return max
end

function syntable.min(table)
	local min

	for a = 1, #table do
		if type(table[a]) == "table" then
			local returnedMin = syntable.min(table[a])

			if not min or returnedMin < min then
				min = returnedMin
			end
		else
			if not min or table[a] < min then
				min = table[a]
			end
		end
	end

	return min
end

function syntable.add(table, number)
	for a = 1, #table do
		if type(table[a]) == "table" then
			table[a] = syntable.add(table[a], number)
		else
			table[a] = table[a] + number
		end
	end

	return table
end

function syntable.multiply(table, number)
	for a = 1, #table do
		if type(table[a]) == "table" then
			table[a] = syntable.multiply(table[a], number)
		else
			table[a] = table[a] * number
		end
	end

	return table
end

function syntable.divide(table, number)
	for a = 1, #table do
		if type(table[a]) == "table" then
			table[a] = syntable.divide(table[a], number)
		else
			table[a] = table[a] / number
		end
	end

	return table
end

function syntable.new(dimensions, defaultvalue, index)
	index = index or 1 -- if index is nil it will be set to 1

	local table = {}

	if index == #dimensions then
		for a = 1, dimensions[index] do
			table[a] = defaultvalue
		end
	else
		for a = 1, dimensions[index] do
			table[a] = syntable.new(dimensions, defaultvalue, index + 1)
		end
	end

	return table
end

function syntable.toString(table, format, indent)
	local tableStr, indentStr = "{", ""

	indent = indent or 1

	if format then
		indentStr = string.rep("    ", indent)
		tableStr = tableStr .. "\n"
	end
  
	for i, v in pairs(table) do
		tableStr = tableStr .. indentStr .. i .. " = "

		local valueType = type(v)

		if valueType == "table" then
			tableStr = tableStr .. syntable.toString(v, format, indent + 1)
		elseif valueType == "string" then
			tableStr = tableStr .. string.format("'%s'", v)
		else
			if v == 0 / 0 then
				tableStr = tableStr .. "0 / 0"
			elseif v == math.huge then
				tableStr = tablestr .. "math.huge"
			else
				tableStr = tableStr .. tostring(v)
			end
		end

		tableStr = tableStr .. ", "

		if format then
			tableStr = tableStr .. "\n"
		end
	end

	if tableStr:sub(#tableStr, #tableStr) == "\n" then
		tableStr = tableStr:sub(1, #tableStr - 3) .. "\n"
	else
		tableStr = tableStr:sub(1, #tableStr - 2)
	end

	if format then
		tableStr = tableStr .. string.rep("    ", indent - 1)
	end

	return tableStr .. "}"
end

return syntable

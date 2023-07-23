--[[
	https://github.com/x-xxoa/synapsea
	core/syndebug.lua

	MIT License
]]--

local syndebug = {
	assertType,
	assertSize,
	assertNanOrInf,
	assertPositive,
	assertNegative,
	assertEqual,
	assertNear,
	assertGreater,
	assertGreaterOrEqual,
	assertLess,
	assertLessOrEqual,
	assertTableType,
	assertEqualTable,
	assertNearTable,
	assertGreaterTable,
	assertGreaterOrEqualTable,
	assertLessTable,
	assertLessOrEqualTable
}

function syndebug.assertType(variable, expectedType)
	if type(variable) ~= expectedType then
		return false
	end

	return true
end

function syndebug.assertSize(table, expectedSize)
	if #table ~= expectedSize then
		return false
	end

	return true
end

function syndebug.assertNanOrInf(table)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertNanOrInf(table[a]) then
				return false
			end
		else
			if table[a] ~= table[a] or table[a] == math.huge then
				return false
			end
		end
	end

	return true
end

function syndebug.assertPositive(table)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertPositive(table[a]) then
				return false
			end
		else
			if table[a] < 0 then
				return false
			end
		end
	end

	return true
end

function syndebug.assertNegative(table)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertNegative(table[a]) then
				return false
			end
		else
			if table[a] > 0 then
				return false
			end
		end
	end

	return true
end

function syndebug.assertEqual(table, expectedValue, recursive)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" and recursive then
			if not syndebug.assertEqual(table[a], expectedValue) then
				return false
			end
		else
			if table[a] ~= expectedValue then
				return false
			end
		end
	end

	return true
end

function syndebug.assertNear(table, expectedValue, allowedDifference)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertNear(table[a], expectedValue, allowedDifference) then
				return false
			end
		else
			if math.abs(table[a] - expectedValue) > allowedDifference then
				return false
			end
		end
	end

	return true
end

function syndebug.assertGreater(table, amount)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertGreater(table[a], amount) then
				return false
			end
		else
			if table[a] <= amount then
				return false
			end
		end
	end

	return true
end

function syndebug.assertGreaterOrEqual(table, amount)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertGreaterOrEqual(table[a], amount) then
				return false
			end
		else
			if table[a] < amount then
				return false
			end
		end
	end

	return true
end

function syndebug.assertLess(table, amount)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertLess(table[a], amount) then
				return false
			end
		else
			if table[a] >= amount then
				return false
			end
		end
	end

	return true
end

function syndebug.assertLessOrEqual(table, amount)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertNegativeOrEqual(table[a], amount) then
				return false
			end
		else
			if table[a] > amount then
				return false
			end
		end
	end

	return true
end

function syndebug.assertTableType(table, expectedType)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertTableType(table[a], expectedType) then
				return false
			end
		else
			if type(table[a]) ~= expectedType then
				return false
			end
		end
	end

	return true
end

function syndebug.assertEqualTable(table, expectedTable)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" and type(expectedTable[a]) == "table" then
			if not syndebug.assertEqualTable(table[a], expectedTable[a]) then
				return false
			end
		else
			if type(table[a]) ~= expectedTable[a] then
				return false
			end
		end
	end

	return true
end

function syndebug.assertNearTable(table, expectedTable, allowedDifference)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertNearTable(table[a], expectedTable[a], allowedDifference) then
				return false
			end
		else
			if math.abs(table[a] - expectedTable[a]) > allowedDifference then
				return false
			end
		end
	end

	return true
end

function syndebug.assertGreaterTable(table, expectedTable)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertGreaterTable(table[a], expectedTable[a]) then
				return false
			end
		else
			if table[a] <= expectedTable[a] then
				return false
			end
		end
	end

	return true
end

function syndebug.assertGreaterOrEqualTable(table, expectedTable)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertGreaterOrEqualTable(table[a], expectedTable[a]) then
				return false
			end
		else
			if table[a] < expectedTable[a] then
				return false
			end
		end
	end

	return true
end

function syndebug.assertLessTable(table, expectedTable)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertLessTable(table[a], expectedTable[a]) then
				return false
			end
		else
			if table[a] >= expectedTable[a] then
				return false
			end
		end
	end

	return true
end

function syndebug.assertLessOrEqualTable(table, expectedTable)
	for a, _ in pairs(table) do
		if type(table[a]) == "table" then
			if not syndebug.assertLessOrEqualTable(table[a], expectedTable[a]) then
				return false
			end
		else
			if table[a] > expectedTable[a] then
				return false
			end
		end
	end

	return true
end

return syndebug

--[[
	https://github.com/x-xxoa/synapsea
	core/array/core/data.lua

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

local dataModule = {
	tableToString,
	arrayToString,
	exportTable,
	importTable
}

function dataModule.tableToString(tbl, format, showFunctions, indent)
	local tableStr, indentStr, equalsStr = "{", "", "="
	indent = indent or 1

	if format then
		indentStr = "    "
		tableStr = tableStr .. "\n"
		equalsStr = " = "
	end

	for i, v in pairs(tbl) do
		local valueType = type(v)

		if showFunctions and valueType == "function" or valueType ~= "function" then
			tableStr = tableStr .. string.rep(indentStr, indent) .. i .. " = "

			if valueType == "number" then
				tableStr = tableStr .. v
			elseif valueType == "table" then
				tableStr = tableStr .. dataModule.tableToString(v, format, showFunctions, indent + 1)
			elseif valueType == "string" then
				tableStr = tableStr .. string.format("\"%s\"", v)
			else
				if v == 0 / 0 then
					tableStr = tableStr .. "0 / 0"
				elseif v == math.huge then
					tableStr = tableStr .. "math.huge"
				else
					tableStr = tableStr .. string.format("\"%s\"", tostring(v))
				end
			end

			tableStr = tableStr .. ","

			if format then
				tableStr = tableStr .. "\n"
			end
		end
	end

	if tableStr:sub(#tableStr, #tableStr) == "," then
		tableStr = tableStr:sub(1, #tableStr - 1)
	end

	if format then
		tableStr = tableStr .. string.rep(indentStr, indent - 1)
	end

	if useBrackets then
		return tableStr .. "]"
	end

	return tableStr .. "}"
end

function dataModule.arrayToString(tbl, useBrackets)
	local tableStr, indentStr = "{", ""

	if useBrackets then
		tableStr = "["
	end

	for a = 1, #tbl do
		local valueType = type(tbl[a])

		if valueType == "number" then
			tableStr = tableStr .. tbl[a]
		elseif valueType == "table" then
			tableStr = tableStr .. dataModule.arrayToString(tbl[a], useBrackets)
		elseif valueType == "string" then
			tableStr = tableStr .. string.format("\"%s\"", tbl[a])
		else
			if v == 0 / 0 then
				tableStr = tableStr .. "0 / 0"
			elseif v == math.huge then
				tableStr = tableStr .. "math.huge"
			else
				tableStr = tableStr .. string.format("\"%s\"", tostring(tbl[a]))
			end
		end

		tableStr = tableStr .. ","

		if format then
			tableStr = tableStr .. "\n"
		end
	end

	if tableStr:sub(#tableStr, #tableStr) == "," then
		tableStr = tableStr:sub(1, #tableStr - 1)
	end

	if useBrackets then
		return tableStr .. "]"
	end

	return tableStr .. "}"
end

function dataModule.exportTable(tbl, fileName, format, showFunctions)
	local f, err, code = io.open(filename, "w")

	assert(f, string.format("Failed opening %s for writing, %s (%s).", fileName, err, code))

	f:write("return ")
	f:write(dataModule.tableToString(tbl, format, showFunctions))

	f:close()
end

function dataModule.exportArray(tbl, fileName)
	local f, err, code = io.open(filename, "w")

	assert(f, string.format("Failed opening %s for writing, %s (%s).", fileName, err, code))

	f:write("return ")
	f:write(dataModule.arrayToString(tbl))

	f:close()
end

function dataModule.importTable(tbl, fileName)
	return dofile(fileName)
end

return dataModule
